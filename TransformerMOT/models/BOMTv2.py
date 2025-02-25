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
from TransformerMOT.models.rp_flooding import RPFlooding


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class BOMT(nn.Module):
    """
    Bearing-Only Multi-Object Tracking Transformer (BOMT) is a model adapted from MT3v2
    https://github.com/JulianoLagana/MT3v2

    The BOMT model will assuming moving sensors & targets (plural) and the measurements are BEARING-ONLY.
    Range and Doppler will be omitted.

    This version will implement RP flooding with false detection

    Normalisation based on playground dimension will be done here.
    """

    __version__ = 2.0

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

        self.measurement_normalization_factor = torch.tensor(
            [
                params.data_generation.dimension[0][1]
                - params.data_generation.dimension[0][0],
                params.data_generation.dimension[0][1]
                - params.data_generation.dimension[0][0],
                1,
                1,
            ]
        ).to(self.params.training.device)
        # position, velocity, angle, d
        # Make sure not to normalise d as it will be normalized together with position

        self.fov_rescaling_factor = self.measurement_normalization_factor[0] * 4
        self.starting_dimension = torch.tensor(params.data_generation.dimension[0][0])

        self.false_detect_embedding = (
            nn.Embedding(1, params.arch.d_model)
            if params.arch.false_detect_embedding
            else None
        )
        self.rp_flooding = RPFlooding(params)

        self.preprocesser = PreProccessor(
            d_model=self.params.arch.d_model,
            d_detections=self.params.arch.d_detections + 1,  # Include d
            normalization_constant=self.measurement_normalization_factor,
        )

        # False detection will not be present for Version 1
        # encoder_norm = nn.LayerNorm(normalized_shape=self.params.arch.d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.params.arch.d_model,
            nhead=self.params.arch.encoder.n_heads,
            dim_feedforward=self.params.arch.encoder.dim_feedforward,
            dropout=self.params.arch.encoder.dropout,
            activation="relu",
            normalize_before=False,
            false_detect_embedding=self.false_detect_embedding,
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

        if self.params.loss.false_classifier:
            self.false_classifier = MLP(
                params.arch.d_model,
                hidden_dim=params.arch.d_prediction_hidden,
                output_dim=1,
                num_layers=1,
            )

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

        sigmoid_measurements = normalized_measurements.sigmoid()
        logits_measurements = torch.log(
            sigmoid_measurements / (1 - sigmoid_measurements)
        )

        # Mask embeddings of measurements that are actually just padding
        masked_embeddings = embeddings
        masked_embeddings = masked_embeddings.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )

        # Project embeddings
        projected_embeddings = self.enc_output_norm(self.enc_output(masked_embeddings))
        return projected_embeddings, logits_measurements

    def get_two_stage_proposals(self, measurement_batch, mask, embeddings):
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
        measurements = measurement_batch[:, :, : self.params.arch.d_detections + 1]
        # (sensor xy, azimuth, d, t)

        # Compute xy position of the measurements using range and azimuth
        xs = (
            measurements[:, :, 3] * (measurements[:, :, 2].cos())
            + measurements[:, :, 0]
        )
        ys = (
            measurements[:, :, 3] * (measurements[:, :, 2].sin())
            + measurements[:, :, 1]
        )

        xy_measurements = torch.stack([xs, ys], 2)

        normalized_xy_meas = xy_measurements / self.fov_rescaling_factor + 0.5

        # Compute projected encoder memory + presigmoid normalized measurements (filtered using the masks)
        result = self.gen_encoder_output_proposals(
            embeddings.permute(1, 0, 2), mask, normalized_xy_meas
        )
        projected_embeddings, normalized_meas_presigmoid = result

        # Compute scores and adjustments
        scores = self.decoder.obj_classifier[self.decoder.num_layers](
            projected_embeddings
        )
        scores = scores.masked_fill(
            mask.unsqueeze(-1), -100_000_000
        )  # Set masked predictions to "0" probability
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
        topk_scores_indices = torch.topk(scores[..., 0], self.num_queries, dim=1)[1]
        repeated_indices = topk_scores_indices.unsqueeze(-1).repeat(
            (1, 1, adjusted_normalized_meas_presigmoid.shape[2])
        )
        topk_adjusted_normalized_meas_presigmoid = torch.gather(
            adjusted_normalized_meas_presigmoid, 1, repeated_indices
        ).detach()
        topk_adjusted_normalized_meas = (
            topk_adjusted_normalized_meas_presigmoid.sigmoid().permute(1, 0, 2)
        )
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
        self,
        measurements: NestedTensor,
        target_coordinates=None,
        unique_id=None,
    ):  # NestedTensor consist of Tensor and mask

        # RP Flooding
        rp_measurements, optim_indices = self.rp_flooding.forward(
            src=measurements.tensors,
            mask=measurements.mask,
            unique_id=unique_id,
            target_coordinates=target_coordinates,
        )  # (B, t * d, feature_dim + 1)
        rp_measurements.to(self.params.training.device)

        # Time encoding
        mapped_time_idx = torch.round(
            rp_measurements[:, :, -1] / self.params.data_generation.interval
        )

        time_encoding = self.temporal_encoder(mapped_time_idx.long())

        if optim_indices is not None:
            optim_indices.to(self.params.training.device)

        mask = measurements.mask = measurements.mask.repeat_interleave(
            repeats=self.rp_flooding.d_radius, dim=-1
        )

        # Preprocessing
        preprocessed_measurements = self.preprocesser(
            rp_measurements[:, :, : self.params.arch.d_detections + 1]  # Include d now
        )

        batch_size, num_batch_max_meas, d_detections = preprocessed_measurements.shape
        preprocessed_measurements = preprocessed_measurements.permute(1, 0, 2)
        time_encoding = time_encoding.permute(1, 0, 2)

        # Feed measurements through encoder
        embeddings = self.encoder(
            preprocessed_measurements, src_key_padding_mask=mask, pos=time_encoding
        )

        # Everything is ok up till here

        aux_classifications = {}

        # loss.contrastive_classifier
        contrastive_classifications = self.contrastive_classifier(
            embeddings.permute(1, 0, 2), padding_mask=mask
        )
        aux_classifications["contrastive_classifications"] = contrastive_classifications

        # False classifications
        if self.params.loss.false_classifier:
            false_classifications = self.false_classifier(embeddings)
            aux_classifications["false_classifications"] = false_classifications

        # 2 Stage / Selection Mechanism
        (
            object_queries,
            query_positional_encodings,
            topk_adjusted_normalized_meas,
            scores,
            adjustments,
            adjusted_normalized_meas,
        ) = self.get_two_stage_proposals(rp_measurements, mask, embeddings)

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

        # Un-normalize state predictions
        intermediate_state_predictions = intermediate_state_predictions_normalized - 0.5
        intermediate_state_predictions *= self.fov_rescaling_factor

        prediction = Prediction(
            positions=intermediate_state_predictions[-1][
                :, :, : self.prediction_space_dimensions
            ],
            velocities=intermediate_state_predictions[-1][
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
            optim_indices,
        )
