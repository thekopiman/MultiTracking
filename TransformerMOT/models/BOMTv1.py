import torch
from torch import nn
from TransformerMOT.modules.position_encoder import LearnedPositionEncoder
from TransformerMOT.modules.mlp import MLP
from TransformerMOT.modules.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    PreProccessor,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from TransformerMOT.modules.contrastive_classifier import ContrastiveClassifier
from TransformerMOT.util.misc import NestedTensor, Prediction
import copy
import math
import numpy as np

from TransformerMOT.models.feature_extractor_encoder import (
    RangeParameterizationLayer,
    FeatureExtractorEncoder,
)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class BOMT(nn.Module):
    """
    Bearing-Only Multi-Object Tracking Transformer (BOMT) is a model adapted from MT3v2
    https://github.com/JulianoLagana/MT3v2

    The BOMT model will assuming moving sensors & targets (plural) and the measurements are BEARING-ONLY.
    Range and Doppler will be omitted.

    This version will not include Refactoring. Refactoring will be done on version 2.

    Features omitted:
    - False detect
    - Refactoring
    - Normalisation of dataset. (It will be done directly on the dataset itself)
    """

    __version__ = 1.0

    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.params = params

        if self.params.training.device == "auto":
            self.params.training.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.temporal_encoder = LearnedPositionEncoder(
            n_timesteps=self.params.data_generation.truncation,
            d_model=self.params.arch.d_model,
        )

        # d_detections is either dim(x,y,azimuth) or dim(x,y,z,azimuth,elevation)
        self.prediction_space_dimensions = (
            self.params.arch.d_detections // 2 + 1
        )  # cartesian (x,y,z) position and velocity

        # Normalization will not occur here.
        # We will do preprocessing on the dataset to set the max dimensions of our simulation. Then this will be used for the normalisation.

        self.measurement_normalization_factor = torch.tensor(
            np.ones(self.prediction_space_dimensions),
            device=torch.device(self.params.training.device),
        )  # This is just a placeholder with ones.
        # Scaling factors will be considered in version 2

        self.rf_layer = RangeParameterizationLayer(self.params.arch.rp_encoder.d_num)
        self.feature_extraction_encoder = FeatureExtractorEncoder(
            d_input=self.params.arch.rp_encoder.d_num,
            d_model=self.params.arch.d_model,
            num_heads=self.params.arch.rp_encoder.n_heads,
            num_layers=self.params.arch.rp_encoder.n_layers,
            dim_feedforward=self.params.arch.rp_encoder.dim_feedforward,
            d_detection=self.params.arch.d_detections,
        )

        self.preprocesser = PreProccessor(
            d_model=self.params.arch.d_model,
            d_detections=self.prediction_space_dimensions,
            normalization_constant=self.measurement_normalization_factor,
        )

        # False detection will not be present for Version 1
        encoder_layer = TransformerEncoderLayer(
            d_model=self.params.arch.d_model,
            nhead=self.params.arch.encoder.n_heads,
            dim_feedforward=self.params.arch.encoder.dim_feedforward,
            dropout=self.params.arch.encoder.dropout,
            activation="relu",
            normalize_before=True,  # We will normalise here to account for the lack of normalisation beforehand
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.params.arch.encoder.n_layers,
            norm=None,
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=self.params.arch.d_model,
            nhead=self.params.arch.decoder.n_heads,
            dim_feedforward=self.params.arch.decoder.dim_feedforward,
            dropout=self.params.arch.decoder.dropout,
            activation="relu",
            normalize_before=True,  # We will normalise here to account for the lack of normalisation beforehand
        )
        decoder_norm = nn.LayerNorm(normalized_shape=self.params.arch.d_model)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.params.arch.decoder.n_layers,
            norm=decoder_norm,
            with_state_refine=False,  # Idk what's this for now
        )
        self.query_embed = nn.Embedding(
            self.params.arch.num_queries, self.params.arch.d_model
        )

        # Create pos/vel delta predictor and existence probability predictor
        self.pos_vel_predictor = MLP(
            self.params.arch.d_model,
            hidden_dim=self.params.arch.d_prediction_hidden,
            output_dim=self.prediction_space_dimensions * 2,
            num_layers=self.params.arch.n_prediction_layers,
        )
        self.uncertainty_predictor = MLP(
            self.params.arch.d_model,
            hidden_dim=self.params.arch.d_prediction_hidden,
            output_dim=self.prediction_space_dimensions * 2,
            num_layers=self.params.arch.n_prediction_layers,
            softplus_at_end=True,
        )
        self.obj_classifier = nn.Linear(self.params.arch.d_model, 1)

        self.return_intermediate = True
        # if self.params.loss.contrastive_classifier:
        if True:
            self.contrastive_classifier = ContrastiveClassifier(params)

        # if self.params.loss.false_classifier:
        #     self.false_classifier = MLP(
        #         params.arch.d_model,
        #         hidden_dim=params.arch.d_prediction_hidden,
        #         output_dim=1,
        #         num_layers=1,
        #     )

        self.two_stage = True
        self.d_model = self.params.arch.d_model

        self._reset_parameters()

        # Initialize delta predictions to zero
        nn.init.constant_(self.pos_vel_predictor.layers[-1].weight.data, 0)
        nn.init.constant_(self.pos_vel_predictor.layers[-1].bias.data, 0)

        # Clone prediction heads for all layers of the decoder (+1 for encoder if two-stage)
        num_pred = (
            (self.decoder.num_layers + 1) if self.two_stage else self.decoder.num_layers
        )
        self.obj_classifier = _get_clones(self.obj_classifier, num_pred)
        self.pos_vel_predictor = _get_clones(self.pos_vel_predictor, num_pred)
        self.uncertainty_predictor = _get_clones(self.uncertainty_predictor, num_pred)
        self.decoder.pos_vel_predictor = self.pos_vel_predictor
        self.decoder.uncertainty_predictor = self.uncertainty_predictor
        self.decoder.obj_classifier = self.obj_classifier

        if self.two_stage:
            # hack implementation for two-stage
            self.enc_output = nn.Linear(self.d_model, self.d_model)
            self.enc_output_norm = nn.LayerNorm(self.d_model)

            # *2 as one is for object query and another for query position encoding
            self.pos_trans = nn.Linear(self.d_model, self.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(self.d_model * 2)

            self.num_queries = self.params.arch.num_queries
        else:
            assert False, "self.two_stage should be = True for now"
            # self.reference_points_linear = nn.Linear(
            #     d_model, self.prediction_space_dimensions * 2
            # )
            # nn.init.xavier_uniform_(self.reference_points_linear.weight.data, gain=1.0)
            # nn.init.constant_(self.reference_points_linear.bias.data, 0.0)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def gen_encoder_output_proposals(
        self, embeddings, memory_padding_mask, normalized_measurements
    ):
        # Compute presigmoid version of normalized measurements
        normalized_measurements_presigmoid = torch.log(
            normalized_measurements / (1 - normalized_measurements)
        )

        # Set to inf invalid measurements (masked or outside the FOV)
        output_proposals_valid = (
            (normalized_measurements > 0.01) & (normalized_measurements < 0.99)
        ).all(-1, keepdim=True)
        normalized_measurements_presigmoid = (
            normalized_measurements_presigmoid.masked_fill(
                memory_padding_mask.unsqueeze(-1), float("inf")
            )
        )
        normalized_measurements_presigmoid = (
            normalized_measurements_presigmoid.masked_fill(
                ~output_proposals_valid, float("inf")
            )
        )

        # Mask embeddings of measurements that are actually just padding
        masked_embeddings = embeddings
        masked_embeddings = masked_embeddings.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        masked_embeddings = masked_embeddings.masked_fill(
            ~output_proposals_valid, float(0)
        )

        # Project embeddings
        projected_embeddings = self.enc_output_norm(self.enc_output(masked_embeddings))
        return projected_embeddings, normalized_measurements_presigmoid

    def get_two_stage_proposals(self, measurement_batch, embeddings):
        """
        Given a batch of measurements and their corresponding embeddings (computed by the encoder), this generates the
        object queries to be fed by the decoder, using the selection mechanism as explained in https://arxiv.org/abs/2104.00734

        @param measurement_batch: Batch of measurements, including their masks.
        @param embeddings: Embeddings computed by the encoder for each of the measurements.
        @return:
            object_queries: queries to be fed to the decoder.
            query_positional_encodings: positional encodings to be added to the object queries.
            reference_points: 2D position estimates to be used as starting points for iterative refinement in the
                decoder.
            enc_outputs_class: predicted existence probability for each measurement.
            enc_outputs_state: predicted adjustment delta for each measurement (measurements are adjusted by summing
                their corresponding deltas before using them as starting points for iterative refinement.
            enc_outputs_coord_unact: adjusted measurements using their corresponding predicted deltas.
        """
        n_measurements, _, c = embeddings.shape
        measurements = measurement_batch.tensors[
            :, :, : self.prediction_space_dimensions
        ]

        # Compute projected encoder memory + presigmoid normalized measurements (filtered using the masks)
        result = self.gen_encoder_output_proposals(
            embeddings.permute(1, 0, 2), measurement_batch.mask, measurements
        )
        projected_embeddings, normalized_meas_presigmoid = result

        # Compute scores and adjustments
        scores = self.decoder.obj_classifier[self.decoder.num_layers](
            projected_embeddings
        )
        scores = scores.masked_fill(
            measurement_batch.mask.unsqueeze(-1), -100_000_000
        )  # Set masked predictions to "0" probability

        # delta
        adjustments = self.pos_vel_predictor[self.decoder.num_layers](
            projected_embeddings
        )

        # Concatenate initial velocity estimates to the measurements
        init_vel_estimates_presigmoid = torch.zeros_like(normalized_meas_presigmoid)
        normalized_meas_presigmoid = torch.cat(
            (
                normalized_meas_presigmoid,
                init_vel_estimates_presigmoid,
            ),
            dim=2,
        )

        # Adjust measurements
        adjusted_normalized_meas_presigmoid = normalized_meas_presigmoid + adjustments
        adjusted_normalized_meas = adjusted_normalized_meas_presigmoid.sigmoid()

        # Select top-k scoring measurements and their corresponding embeddings

        num_queries = min(
            scores.shape[1], self.num_queries
        )  # Account for cases when the input data is too small

        topk_scores_indices = torch.topk(scores[..., 0], num_queries, dim=1)[1]
        repeated_indices = topk_scores_indices.unsqueeze(-1).repeat(
            (1, 1, adjusted_normalized_meas_presigmoid.shape[2])
        )
        topk_adjusted_normalized_meas_presigmoid = torch.gather(
            adjusted_normalized_meas_presigmoid, 1, repeated_indices
        ).detach()
        topk_adjusted_normalized_meas = (
            topk_adjusted_normalized_meas_presigmoid.sigmoid().permute(1, 0, 2)
        )
        # i 1-k
        topk_memory = torch.gather(
            projected_embeddings.detach(),
            1,
            topk_scores_indices.unsqueeze(-1).repeat(1, 1, self.params.arch.d_model),
        )

        # Compute object queries and their positional encodings by feeding the top-k memory through FFN+LayerNorm
        pos_trans_out = self.pos_trans_norm(self.pos_trans(topk_memory))
        query_positional_encodings, object_queries = torch.split(
            pos_trans_out, c, dim=2
        )
        query_positional_encodings = query_positional_encodings.permute(1, 0, 2)
        object_queries = object_queries.permute(1, 0, 2)

        return (
            object_queries,
            query_positional_encodings,
            topk_adjusted_normalized_meas,
            scores,
            adjustments,
            adjusted_normalized_meas,
        )

    def forward(
        self, measurements: NestedTensor
    ):  # NestedTensor consist of Tensor and mask

        # Feed through feature extraction encoder
        measurements_post_FE = self.rf_layer(
            measurements.tensors.permute(0, 2, 1)[:, : self.params.arch.d_detections, :]
        )
        measurements_post_FE = self.feature_extraction_encoder(
            measurements_post_FE
        ).permute(0, 2, 1)

        # Time encoding
        mapped_time_idx = torch.round(
            measurements.tensors[:, :, -1] / self.params.data_generation.interval
        )

        time_encoding = self.temporal_encoder(mapped_time_idx.long())

        # Preprocessing
        preprocessed_measurements = self.preprocesser(
            measurements_post_FE[:, :, : self.prediction_space_dimensions]
        )
        mask = measurements.mask

        batch_size, num_batch_max_meas, d_detections = preprocessed_measurements.shape
        preprocessed_measurements = preprocessed_measurements.permute(1, 0, 2)
        time_encoding = time_encoding.permute(1, 0, 2)

        # Feed measurements through encoder
        embeddings = self.encoder(
            preprocessed_measurements, src_key_padding_mask=mask, pos=time_encoding
        )

        aux_classifications = {}

        # loss.contrastive_classifier
        contrastive_classifications = self.contrastive_classifier(
            embeddings.permute(1, 0, 2), padding_mask=mask
        )
        aux_classifications["contrastive_classifications"] = contrastive_classifications

        # False classifications omitted

        # 2 Stage / Selection Mechanism
        (
            object_queries,
            query_positional_encodings,
            topk_adjusted_normalized_meas,
            scores,
            adjustments,
            adjusted_normalized_meas,
        ) = self.get_two_stage_proposals(measurements, embeddings)

        result = self.decoder(
            object_queries,
            embeddings,
            encoder_embeddings_padding_mask=mask,
            encoder_embeddings_positional_encoding=time_encoding,
            object_queries_positional_encoding=query_positional_encodings,
            reference_points=topk_adjusted_normalized_meas,
        )

        (
            intermediate_state_predictions_normalized,
            intermediate_uncertainties,
            intermediate_logits,
            debug_dict,
        ) = result

        prediction = Prediction(
            positions=intermediate_state_predictions_normalized[-1][
                :, :, : self.prediction_space_dimensions
            ],
            velocities=intermediate_state_predictions_normalized[-1][
                :, :, self.prediction_space_dimensions :
            ],
            uncertainties=intermediate_uncertainties[-1],
            logits=intermediate_logits[-1],
        )
        intermediate_predictions = (
            [
                Prediction(
                    positions=p[:, :, : self.prediction_space_dimensions],
                    velocities=p[:, :, self.prediction_space_dimensions :],
                    uncertainties=u,
                    logits=l,
                )
                for p, l, u in zip(
                    intermediate_state_predictions_normalized[:-1],
                    intermediate_logits[:-1],
                    intermediate_uncertainties[:-1],
                )
            ]
            if self.return_intermediate
            else None
        )
        encoder_prediction = (
            Prediction(
                positions=adjusted_normalized_meas[
                    :, :, : self.prediction_space_dimensions
                ],
                velocities=adjusted_normalized_meas[
                    :, :, self.prediction_space_dimensions :
                ],
                logits=scores,
            )
            if self.two_stage
            else None
        )
        return (
            prediction,
            intermediate_predictions,
            encoder_prediction,
            aux_classifications,
            debug_dict,
        )
