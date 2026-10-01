"""BiLSTM followed by masked multi-head self-attention."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        attention_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last_valid"}:
            raise ValueError("pooling must be 'mean' or 'last_valid'")
        directions = 2 if bidirectional else 1
        output_size = directions * hidden_size
        if output_size % attention_heads != 0:
            raise ValueError(
                f"LSTM output size {output_size} must be divisible by attention_heads={attention_heads}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.pooling = pooling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, num_classes)

    def forward(
        self,
        inputs: Tensor,
        lengths: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if inputs.ndim == 4:
            inputs = inputs.flatten(start_dim=2)
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(f"Expected [B,T,{self.input_size}] or [B,T,J,2], got {tuple(inputs.shape)}")
        batch, time, _ = inputs.shape
        if lengths is None:
            if padding_mask is None:
                lengths = torch.full((batch,), time, dtype=torch.long, device=inputs.device)
            else:
                lengths = (~padding_mask).sum(dim=1).to(dtype=torch.long)
        lengths = lengths.to(device=inputs.device, dtype=torch.long).clamp(min=1, max=time)
        if padding_mask is None:
            padding_mask = torch.arange(time, device=inputs.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            padding_mask = padding_mask.to(device=inputs.device, dtype=torch.bool)

        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=time)
        attended, attention_weights = self.attention(
            lstm_output,
            lstm_output,
            lstm_output,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
        )
        valid = (~padding_mask).unsqueeze(-1).to(attended.dtype)
        if self.pooling == "mean":
            pooled = (attended * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            last_indices = lengths - 1
            pooled = attended[torch.arange(batch, device=inputs.device), last_indices]
        logits = self.classifier(self.dropout(pooled))
        if return_attention:
            return logits, attention_weights
        return logits


class PaperBiLSTMAttentionClassifier(nn.Module):
    """Paper-protocol architecture kept separate from the E0 baseline.

    The paper-reproduction path adds the residual connection, LayerNorm and
    256 -> 64 projection described in the supplied architecture table. The
    dropout after ReLU is the paper-reported dropout; LSTM and attention
    dropout are independently configurable because the paper does not specify
    them.
    """

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        attention_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        lstm_internal_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual: bool = True,
        layer_norm: bool = True,
        projection_size: int = 64,
    ) -> None:
        super().__init__()
        directions = 2 if bidirectional else 1
        output_size = directions * hidden_size
        if output_size % attention_heads != 0:
            raise ValueError(
                f"LSTM output size {output_size} must be divisible by attention_heads={attention_heads}"
            )
        if projection_size <= 0:
            raise ValueError("projection_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.projection_size = projection_size
        self.residual = residual
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_internal_dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else nn.Identity()
        self.projection = nn.Linear(output_size, projection_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(projection_size, num_classes)

    def forward(
        self,
        inputs: Tensor,
        lengths: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if inputs.ndim == 4:
            inputs = inputs.flatten(start_dim=2)
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(f"Expected [B,T,{self.input_size}] or [B,T,J,2], got {tuple(inputs.shape)}")
        batch, time, _ = inputs.shape
        if lengths is None:
            if padding_mask is None:
                lengths = torch.full((batch,), time, dtype=torch.long, device=inputs.device)
            else:
                lengths = (~padding_mask).sum(dim=1).to(dtype=torch.long)
        lengths = lengths.to(device=inputs.device, dtype=torch.long).clamp(min=1, max=time)
        if padding_mask is None:
            padding_mask = torch.arange(time, device=inputs.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            padding_mask = padding_mask.to(device=inputs.device, dtype=torch.bool)

        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bilstm(packed)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=time)
        attended, attention_weights = self.attention(
            lstm_output,
            lstm_output,
            lstm_output,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
        )
        fused = lstm_output + attended if self.residual else attended
        fused = self.layer_norm(fused)
        valid = (~padding_mask).unsqueeze(-1).to(fused.dtype)
        pooled = (fused * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        projected = self.activation(self.projection(pooled))
        logits = self.classifier(self.dropout(projected))
        if return_attention:
            return logits, attention_weights
        return logits


class KinematicBiLSTMAttentionClassifier(nn.Module):
    """BiLSTM + Attention augmented with Joint Velocity and Key Bone Relational features.

    Inputs can be [B, T, 25, 2] or [B, T, 50].
    Extracts:
      1. Position (50 dims)
      2. Joint velocity Δx, Δy (50 dims)
      3. Key relational bone/joint distances (10 dims):
         - Left/Right wrist Euclidean distance (crucial for clapping vs drumming)
         - Left/Right elbow Euclidean distance
         - Wrists to shoulders distances (L-L, R-R)
         - Wrists to hips distances (L-L, R-R)
         - Ankles distance (L-R)
         - Knees distance (L-R)
         - Torso height (Neck to MidHip vertical displacement)
         - Body compression (Neck to mean ankles vertical displacement)
      Total raw feature dimension: 110.
    Passes through LayerNorm & projection to match BiLSTM hidden dimensions.
    """

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        attention_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        lstm_internal_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual: bool = True,
        layer_norm: bool = True,
        projection_size: int = 64,
        raw_feature_dim: int = 110,
    ) -> None:
        super().__init__()
        directions = 2 if bidirectional else 1
        output_size = directions * hidden_size
        if output_size % attention_heads != 0:
            raise ValueError(
                f"LSTM output size {output_size} must be divisible by attention_heads={attention_heads}"
            )
        self.raw_feature_dim = raw_feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.projection_size = projection_size
        self.residual = residual

        self.feature_norm = nn.LayerNorm(raw_feature_dim) if layer_norm else nn.Identity()
        self.bilstm = nn.LSTM(
            input_size=raw_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_internal_dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else nn.Identity()
        self.projection = nn.Linear(output_size, projection_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(projection_size, num_classes)

    def extract_kinematic_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 3 and inputs.shape[-1] == 50:
            joints = inputs.view(inputs.shape[0], inputs.shape[1], 25, 2)
            pos = inputs
        elif inputs.ndim == 4 and inputs.shape[2:] == (25, 2):
            joints = inputs
            pos = inputs.flatten(start_dim=2)
        else:
            raise ValueError(f"Expected [B,T,50] or [B,T,25,2], got {tuple(inputs.shape)}")

        # 1. Temporal joint velocity Δx, Δy
        vel = torch.zeros_like(joints)
        vel[:, 1:] = joints[:, 1:] - joints[:, :-1]
        vel_flat = vel.flatten(start_dim=2)

        # 2. Key relational distances
        # 4: RWrist, 7: LWrist; 3: RElbow, 6: LElbow; 2: RShoulder, 5: LShoulder
        # 9: RHip, 12: LHip; 10: RKnee, 13: LKnee; 11: RAnkle, 14: LAnkle
        # 1: Neck, 8: MidHip
        wrist_dist = torch.norm(joints[:, :, 7] - joints[:, :, 4], dim=-1, keepdim=True)
        elbow_dist = torch.norm(joints[:, :, 6] - joints[:, :, 3], dim=-1, keepdim=True)
        lwrist_lsh = torch.norm(joints[:, :, 7] - joints[:, :, 5], dim=-1, keepdim=True)
        rwrist_rsh = torch.norm(joints[:, :, 4] - joints[:, :, 2], dim=-1, keepdim=True)
        lwrist_lhip = torch.norm(joints[:, :, 7] - joints[:, :, 12], dim=-1, keepdim=True)
        rwrist_rhip = torch.norm(joints[:, :, 4] - joints[:, :, 9], dim=-1, keepdim=True)
        ankle_dist = torch.norm(joints[:, :, 14] - joints[:, :, 11], dim=-1, keepdim=True)
        knee_dist = torch.norm(joints[:, :, 13] - joints[:, :, 10], dim=-1, keepdim=True)
        torso_h = torch.abs(joints[:, :, 1, 1:2] - joints[:, :, 8, 1:2])
        ankle_mid_y = 0.5 * (joints[:, :, 11, 1:2] + joints[:, :, 14, 1:2])
        body_comp = torch.abs(joints[:, :, 1, 1:2] - ankle_mid_y)

        if self.raw_feature_dim == 125:
            # Focused end-effector dynamics (no noise on occluded toes/ears)
            wrist_dx = torch.abs(joints[:, :, 7, 0:1] - joints[:, :, 4, 0:1])
            wrist_dy = torch.abs(joints[:, :, 7, 1:2] - joints[:, :, 4, 1:2])

            rw_vel = vel[:, :, 4]
            lw_vel = vel[:, :, 7]
            rel_wrist_vel = rw_vel - lw_vel
            wrist_vel_dot = (rw_vel * lw_vel).sum(dim=-1, keepdim=True)
            rw_speed = torch.norm(rw_vel + 1e-6, dim=-1, keepdim=True)
            lw_speed = torch.norm(lw_vel + 1e-6, dim=-1, keepdim=True)

            rw_acc = torch.zeros_like(rw_vel)
            lw_acc = torch.zeros_like(lw_vel)
            rw_acc[:, 1:] = rw_vel[:, 1:] - rw_vel[:, :-1]
            lw_acc[:, 1:] = lw_vel[:, 1:] - lw_vel[:, :-1]
            rw_acc_mag = torch.norm(rw_acc + 1e-6, dim=-1, keepdim=True)
            lw_acc_mag = torch.norm(lw_acc + 1e-6, dim=-1, keepdim=True)

            r_upper = joints[:, :, 3] - joints[:, :, 2]
            r_fore = joints[:, :, 4] - joints[:, :, 3]
            r_upper_norm = torch.norm(r_upper, dim=-1, keepdim=True).clamp_min(1e-6)
            r_fore_norm = torch.norm(r_fore, dim=-1, keepdim=True).clamp_min(1e-6)
            r_elbow_cos = ((r_upper * r_fore).sum(dim=-1, keepdim=True) / (r_upper_norm * r_fore_norm)).clamp(-0.999, 0.999)

            l_upper = joints[:, :, 6] - joints[:, :, 5]
            l_fore = joints[:, :, 7] - joints[:, :, 6]
            l_upper_norm = torch.norm(l_upper, dim=-1, keepdim=True).clamp_min(1e-6)
            l_fore_norm = torch.norm(l_fore, dim=-1, keepdim=True).clamp_min(1e-6)
            l_elbow_cos = ((l_upper * l_fore).sum(dim=-1, keepdim=True) / (l_upper_norm * l_fore_norm)).clamp(-0.999, 0.999)

            features = torch.cat([
                pos, vel_flat,
                wrist_dist, elbow_dist, lwrist_lsh, rwrist_rsh,
                lwrist_lhip, rwrist_rhip, ankle_dist, knee_dist,
                torso_h, body_comp,
                wrist_dx, wrist_dy,
                rel_wrist_vel, wrist_vel_dot, rw_speed, lw_speed,
                rw_acc, lw_acc, rw_acc_mag, lw_acc_mag,
                r_elbow_cos, l_elbow_cos
            ], dim=-1)
            return features

        if self.raw_feature_dim == 167:
            # 3. Joint acceleration (Δ²x, Δ²y)
            acc = torch.zeros_like(vel)
            acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
            acc_flat = acc.flatten(start_dim=2)

            # 4. Decomposed wrist distance (dx, dy)
            wrist_dx = torch.abs(joints[:, :, 7, 0:1] - joints[:, :, 4, 0:1])
            wrist_dy = torch.abs(joints[:, :, 7, 1:2] - joints[:, :, 4, 1:2])

            # 5. Relative wrist velocity vector & dot product (clap vs drum distinction)
            rw_vel = vel[:, :, 4]
            lw_vel = vel[:, :, 7]
            rel_wrist_vel = rw_vel - lw_vel
            wrist_vel_dot = (rw_vel * lw_vel).sum(dim=-1, keepdim=True)

            # 6. Elbow flexion angles (cosine of elbow angle)
            r_upper = joints[:, :, 3] - joints[:, :, 2]
            r_fore = joints[:, :, 4] - joints[:, :, 3]
            r_upper_norm = torch.norm(r_upper, dim=-1, keepdim=True).clamp_min(1e-6)
            r_fore_norm = torch.norm(r_fore, dim=-1, keepdim=True).clamp_min(1e-6)
            r_elbow_cos = ((r_upper * r_fore).sum(dim=-1, keepdim=True) / (r_upper_norm * r_fore_norm)).clamp(-0.999, 0.999)

            l_upper = joints[:, :, 6] - joints[:, :, 5]
            l_fore = joints[:, :, 7] - joints[:, :, 6]
            l_upper_norm = torch.norm(l_upper, dim=-1, keepdim=True).clamp_min(1e-6)
            l_fore_norm = torch.norm(l_fore, dim=-1, keepdim=True).clamp_min(1e-6)
            l_elbow_cos = ((l_upper * l_fore).sum(dim=-1, keepdim=True) / (l_upper_norm * l_fore_norm)).clamp(-0.999, 0.999)

            features = torch.cat([
                pos, vel_flat, acc_flat,
                wrist_dist, elbow_dist, lwrist_lsh, rwrist_rsh,
                lwrist_lhip, rwrist_rhip, ankle_dist, knee_dist,
                torso_h, body_comp,
                wrist_dx, wrist_dy, rel_wrist_vel, wrist_vel_dot,
                r_elbow_cos, l_elbow_cos
            ], dim=-1)
            return features

        features = torch.cat([
            pos, vel_flat,
            wrist_dist, elbow_dist, lwrist_lsh, rwrist_rsh,
            lwrist_lhip, rwrist_rhip, ankle_dist, knee_dist,
            torso_h, body_comp
        ], dim=-1)
        return features

    def forward(
        self,
        inputs: Tensor,
        lengths: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        features = self.extract_kinematic_features(inputs)
        batch, time, _ = features.shape
        if lengths is None:
            if padding_mask is None:
                lengths = torch.full((batch,), time, dtype=torch.long, device=inputs.device)
            else:
                lengths = (~padding_mask).sum(dim=1).to(dtype=torch.long)
        lengths = lengths.to(device=inputs.device, dtype=torch.long).clamp(min=1, max=time)
        if padding_mask is None:
            padding_mask = torch.arange(time, device=inputs.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            padding_mask = padding_mask.to(device=inputs.device, dtype=torch.bool)

        normed = self.feature_norm(features)
        packed = pack_padded_sequence(normed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bilstm(packed)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=time)
        attended, attention_weights = self.attention(
            lstm_output,
            lstm_output,
            lstm_output,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
        )
        fused = lstm_output + attended if self.residual else attended
        fused = self.layer_norm(fused)
        valid = (~padding_mask).unsqueeze(-1).to(fused.dtype)
        pooled = (fused * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        projected = self.activation(self.projection(pooled))
        logits = self.classifier(self.dropout(projected))
        if return_attention:
            return logits, attention_weights
        return logits
