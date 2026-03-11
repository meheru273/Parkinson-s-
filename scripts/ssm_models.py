"""
SSM/Mamba Model Variants for Parkinson's Detection
===================================================
Three model architectures using State-Space Models (Mamba-style selective SSM):

1. PureSSMModel       — SSM replaces CrossAttention entirely
2. SSMEncoderModel    — SSM encoder → CrossAttention → pooling
3. GatedFusionModel   — Parallel CrossAttention + SSM with gated fusion

All models share the same forward(left_wrist, right_wrist, device=None) interface
and get_features(...) method as the base MainModel.

The Mamba block is implemented from scratch in pure PyTorch — no external
mamba-ssm CUDA dependency needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared Components (reused from base model)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        timestep = x.size(1)
        return x + self.pe[:timestep, :].unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)
        return x


class CrossAttention(nn.Module):
    """Bidirectional cross-attention between two streams (from base model)."""

    def __init__(self, model_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attention_1to2 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_2to1 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_1 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_2 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross_1 = nn.LayerNorm(model_dim)
        self.norm_cross_2 = nn.LayerNorm(model_dim)
        self.norm_self_1 = nn.LayerNorm(model_dim)
        self.norm_self_2 = nn.LayerNorm(model_dim)
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)

    def forward(self, channel_1, channel_2):
        # Cross attention
        c1_cross, _ = self.cross_attention_1to2(query=channel_1, key=channel_2, value=channel_2)
        c1_cross = self.norm_cross_1(channel_1 + c1_cross)
        c2_cross, _ = self.cross_attention_2to1(query=channel_2, key=channel_1, value=channel_1)
        c2_cross = self.norm_cross_2(channel_2 + c2_cross)
        # Self attention
        c1_self, _ = self.self_attention_1(query=c1_cross, key=c1_cross, value=c1_cross)
        c1_self = self.norm_self_1(c1_cross + c1_self)
        c2_self, _ = self.self_attention_2(query=c2_cross, key=c2_cross, value=c2_cross)
        c2_self = self.norm_self_2(c2_cross + c2_self)
        # FFN
        return self.feed_forward_1(c1_self), self.feed_forward_2(c2_self)


# =============================================================================
# Mamba / SSM Building Blocks
# =============================================================================

class MambaBlock(nn.Module):
    """
    Selective State-Space Model block (Mamba-style).

    Implements the core Mamba architecture from scratch in pure PyTorch:
      - Input projection with expansion
      - Causal depthwise 1D convolution
      - Selective scan with input-dependent Δ, B, C
      - Output gating and projection

    Args:
        model_dim:  Input/output dimension
        d_state:    SSM state dimension (N in the paper)
        expand:     Expansion factor for inner dimension
        d_conv:     Kernel size for the causal 1D convolution
        dropout:    Dropout rate
    """

    def __init__(
        self,
        model_dim: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.d_state = d_state
        self.d_inner = model_dim * expand
        self.d_conv = d_conv

        # Input projection: project to 2 * d_inner (one for main path, one for gate)
        self.in_proj = nn.Linear(model_dim, self.d_inner * 2, bias=False)

        # Causal depthwise 1D conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters — input-dependent projections
        # x → Δ (controls discretization step size, selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # Δ projection (rank-1 bottleneck → d_inner)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Learnable A parameter (structured as log for stability)
        # Shape: (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection scalar per channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, model_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, model_dim)
        Returns:
            (B, L, model_dim)
        """
        B, L, _ = x.shape

        # Input projection → split into main path and gate
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal 1D conv on the main path
        x_conv = x_main.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal: trim to original length
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameter projections from convolved input
        x_ssm_params = self.x_proj(x_conv)  # (B, L, d_state*2 + 1)
        delta = x_ssm_params[:, :, :1]  # (B, L, 1)
        B_input = x_ssm_params[:, :, 1 : 1 + self.d_state]  # (B, L, d_state)
        C_input = x_ssm_params[:, :, 1 + self.d_state :]  # (B, L, d_state)

        # Discretize delta
        delta = self.dt_proj(delta)  # (B, L, d_inner)
        delta = F.softplus(delta)  # ensure positive

        # Get continuous A
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # Selective scan (sequential for compatibility)
        y = self._selective_scan(x_conv, delta, A, B_input, C_input)

        # Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Output gate (SiLU gate)
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y

    def _selective_scan(self, x, delta, A, B_input, C_input):
        """
        Sequential selective scan implementation.

        Args:
            x:        (B, L, d_inner) — input after conv
            delta:    (B, L, d_inner) — discretization step sizes
            A:        (d_inner, d_state) — state transition matrix (continuous)
            B_input:  (B, L, d_state) — input-dependent B
            C_input:  (B, L, d_state) — input-dependent C

        Returns:
            y: (B, L, d_inner)
        """
        B_batch, L, d_inner = x.shape
        d_state = self.d_state

        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        # delta: (B, L, d_inner) → (B, L, d_inner, 1)
        # A: (d_inner, d_state) → (1, 1, d_inner, d_state)
        delta_A = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, d_inner, d_state)

        delta_B = (
            delta.unsqueeze(-1) * B_input.unsqueeze(2)
        )  # (B, L, d_inner, d_state)

        # Sequential scan
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            # h = A_bar * h + B_bar * x
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
            # y = C * h
            y_t = (h * C_input[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y


class SSMLayer(nn.Module):
    """Single SSM layer: MambaBlock + residual + LayerNorm + FeedForward."""

    def __init__(
        self,
        model_dim: int,
        d_ff: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.mamba = MambaBlock(model_dim, d_state, expand, d_conv, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, d_ff, dropout)

    def forward(self, x):
        # Pre-norm Mamba block with residual
        x = x + self.mamba(self.norm1(x))
        # FFN (already has internal residual + layer norm)
        x = self.ffn(x)
        return x


class SSMStack(nn.Module):
    """Stack of N SSM layers."""

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        d_ff: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SSMLayer(model_dim, d_ff, d_state, expand, d_conv, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Gated Fusion Module
# =============================================================================

class GatedFusion(nn.Module):
    """
    Learned gated fusion between two embedding streams.
    g = σ(W·[x_a; x_b] + b)
    output = g * x_a + (1 - g) * x_b
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, x_a, x_b):
        g = self.gate(torch.cat([x_a, x_b], dim=-1))
        return g * x_a + (1 - g) * x_b


# =============================================================================
# Classifier Head Builder
# =============================================================================

def _make_classifier_head(input_dim, hidden_dim, num_classes, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


# =============================================================================
# Model 1: Pure SSM (replaces CrossAttention entirely)
# =============================================================================

class PureSSMModel(nn.Module):
    """
    Pure state-space model replacing all cross-attention layers.

    Architecture:
        Projection → PositionalEncoding → SSMStack(left) + SSMStack(right)
        → Pool → Concat → ClassifierHeads
    """

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 8,       # unused, kept for interface compatibility
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        timestep: int = 256,
        num_classes: int = 2,
        fusion_method: str = "concat",
        # SSM-specific
        ssm_d_state: int = 16,
        ssm_expand: int = 2,
        ssm_d_conv: int = 4,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.timestep = timestep

        # Projections
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)

        # SSM stacks (one per wrist)
        self.left_ssm = SSMStack(
            num_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )
        self.right_ssm = SSMStack(
            num_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        fusion_dim = model_dim * 2
        self.head_hc_vs_pd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)
        self.head_pd_vs_dd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)

    def get_features(self, left_wrist, right_wrist, device=None):
        left = self.dropout(self.positional_encoding(self.left_projection(left_wrist)))
        right = self.dropout(self.positional_encoding(self.right_projection(right_wrist)))

        left = self.left_ssm(left)
        right = self.right_ssm(right)

        left_pool = self.global_pool(left.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right.transpose(1, 2)).squeeze(-1)

        return torch.cat([left_pool, right_pool], dim=1)

    def forward(self, left_wrist, right_wrist, device=None):
        fused = self.get_features(left_wrist, right_wrist, device)
        return self.head_hc_vs_pd(fused), self.head_pd_vs_dd(fused)


# =============================================================================
# Model 2: SSM Encoder → CrossAttention
# =============================================================================

class SSMEncoderModel(nn.Module):
    """
    SSM encodes temporal patterns first, then CrossAttention enables
    inter-limb interaction.

    Architecture:
        Projection → SSMStack(left/right encoders) → CrossAttention layers
        → Pool → Concat → ClassifierHeads
    """

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        timestep: int = 256,
        num_classes: int = 2,
        fusion_method: str = "concat",
        # SSM-specific
        ssm_d_state: int = 16,
        ssm_expand: int = 2,
        ssm_d_conv: int = 4,
        ssm_num_layers: int = None,  # defaults to num_layers if None
        ca_num_layers: int = None,   # defaults to num_layers if None
    ):
        super().__init__()
        self.model_dim = model_dim
        self.timestep = timestep

        ssm_layers = ssm_num_layers if ssm_num_layers is not None else num_layers
        ca_layers = ca_num_layers if ca_num_layers is not None else num_layers

        # Projections
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)

        # SSM encoder stacks
        self.left_ssm_encoder = SSMStack(
            ssm_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )
        self.right_ssm_encoder = SSMStack(
            ssm_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )

        # CrossAttention layers on SSM-encoded representations
        self.ca_layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout)
            for _ in range(ca_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        fusion_dim = model_dim * 2
        self.head_hc_vs_pd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)
        self.head_pd_vs_dd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)

    def get_features(self, left_wrist, right_wrist, device=None):
        left = self.dropout(self.positional_encoding(self.left_projection(left_wrist)))
        right = self.dropout(self.positional_encoding(self.right_projection(right_wrist)))

        # SSM encoding
        left = self.left_ssm_encoder(left)
        right = self.right_ssm_encoder(right)

        # Cross-attention interaction
        for layer in self.ca_layers:
            left, right = layer(left, right)

        left_pool = self.global_pool(left.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right.transpose(1, 2)).squeeze(-1)

        return torch.cat([left_pool, right_pool], dim=1)

    def forward(self, left_wrist, right_wrist, device=None):
        fused = self.get_features(left_wrist, right_wrist, device)
        return self.head_hc_vs_pd(fused), self.head_pd_vs_dd(fused)


# =============================================================================
# Model 3: Parallel CrossAttention + SSM with Gated Fusion
# =============================================================================

class GatedFusionModel(nn.Module):
    """
    Parallel dual-path model with learned gated fusion.

    Architecture:
        Projection
           ├── CrossAttention stack → Pool → left_ca, right_ca
           └── SSM stack            → Pool → left_ssm, right_ssm
                ↓
           GatedFusion(left_ca, left_ssm), GatedFusion(right_ca, right_ssm)
                ↓
           Concat(left_fused, right_fused)
                ↓
           ClassifierHeads
    """

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        timestep: int = 256,
        num_classes: int = 2,
        fusion_method: str = "concat",
        # SSM-specific
        ssm_d_state: int = 16,
        ssm_expand: int = 2,
        ssm_d_conv: int = 4,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.timestep = timestep

        # Shared projections (input is shared, then forks into two paths)
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)

        # Path A: CrossAttention stack
        self.ca_layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Path B: SSM stacks (one per wrist)
        self.left_ssm = SSMStack(
            num_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )
        self.right_ssm = SSMStack(
            num_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Gated fusion (one per wrist)
        self.left_gate = GatedFusion(model_dim)
        self.right_gate = GatedFusion(model_dim)

        fusion_dim = model_dim * 2
        self.head_hc_vs_pd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)
        self.head_pd_vs_dd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)

    def get_features(self, left_wrist, right_wrist, device=None):
        left = self.dropout(self.positional_encoding(self.left_projection(left_wrist)))
        right = self.dropout(self.positional_encoding(self.right_projection(right_wrist)))

        # ---- Path A: CrossAttention ----
        left_ca, right_ca = left, right
        for layer in self.ca_layers:
            left_ca, right_ca = layer(left_ca, right_ca)
        left_ca_pool = self.global_pool(left_ca.transpose(1, 2)).squeeze(-1)
        right_ca_pool = self.global_pool(right_ca.transpose(1, 2)).squeeze(-1)

        # ---- Path B: SSM ----
        left_ssm = self.left_ssm(left)
        right_ssm = self.right_ssm(right)
        left_ssm_pool = self.global_pool(left_ssm.transpose(1, 2)).squeeze(-1)
        right_ssm_pool = self.global_pool(right_ssm.transpose(1, 2)).squeeze(-1)

        # ---- Gated fusion ----
        left_fused = self.left_gate(left_ca_pool, left_ssm_pool)
        right_fused = self.right_gate(right_ca_pool, right_ssm_pool)

        return torch.cat([left_fused, right_fused], dim=1)

    def forward(self, left_wrist, right_wrist, device=None):
        fused = self.get_features(left_wrist, right_wrist, device)
        return self.head_hc_vs_pd(fused), self.head_pd_vs_dd(fused)
