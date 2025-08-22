import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # Keep as 2D: [max_len, d_model]
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)  # Get sequence length from dimension 1
        # Add positional encoding: [batch_size, seq_len, d_model] + [seq_len, d_model]
        return x + self.pe[:seq_len, :].unsqueeze(0)  # Broadcast across batch dimension


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        # Ensure we have 3D tensors: [batch_size, seq_len, d_model]
        if len(query.shape) != 3:
            raise ValueError(f"Expected 3D tensor, got {query.shape}")
            
        batch_size, seq_len_q, d_model = query.size()
        seq_len_k = key.size(1)
        
        # Project and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_w = F.softmax(scores, dim=-1)
        attention_w = self.dropout(attention_w)
        
        context = torch.matmul(attention_w, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        output = self.w_o(context)
        output = self.layer_norm(output + query)  # Residual connection
        
        return output, attention_w


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)  # Residual connection
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention_1to2 = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_attention_2to1 = MultiheadAttention(d_model, num_heads, dropout)
        self.self_attention_1 = MultiheadAttention(d_model, num_heads, dropout)
        self.self_attention_2 = MultiheadAttention(d_model, num_heads, dropout)
        
        self.feed_forward_1 = FeedForward(d_model, d_ff, dropout)       
        self.feed_forward_2 = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, channel_1, channel_2):
        # Cross attention
        channel_1_cross, _ = self.cross_attention_1to2(query=channel_1, key=channel_2, value=channel_2)
        channel_2_cross, _ = self.cross_attention_2to1(query=channel_2, key=channel_1, value=channel_1)
        
        # Self attention
        channel_1_self, _ = self.self_attention_1(query=channel_1_cross, key=channel_1_cross, value=channel_1_cross)
        channel_2_self, _ = self.self_attention_2(query=channel_2_cross, key=channel_2_cross, value=channel_2_cross)
        
        # Feed forward
        channel_1_out = self.feed_forward_1(channel_1_self)
        channel_2_out = self.feed_forward_2(channel_2_self)

        return channel_1_out, channel_2_out


class DualChannelTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,  
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        seq_len: int = 256,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Project input dimensions to d_model
        self.left_projection = nn.Linear(input_dim, d_model)
        self.right_projection = nn.Linear(input_dim, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        self.layers = nn.ModuleList([
            CrossAttention(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification heads
        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # Binary: HC vs PD
        )
        
        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # Binary: PD vs DD
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, left_wrist, right_wrist):
        # Input shape: [batch_size, seq_len, input_dim]
        
        # Project to model dimension
        left_encoded = self.left_projection(left_wrist)   # [batch, seq_len, d_model]
        right_encoded = self.right_projection(right_wrist) # [batch, seq_len, d_model]

        # Add positional encoding (no transpose needed!)
        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        # Apply dropout
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        # Pass through cross-attention layers
        for layer in self.layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)

        # Global pooling: [batch, seq_len, d_model] -> [batch, d_model]
        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)

        # Fuse features from both channels
        fused_features = torch.cat([left_pool, right_pool], dim=1)  # [batch, d_model*2]
        
        # Classification
        logits_hc_vs_pd = self.head_hc_vs_pd(fused_features)
        logits_pd_vs_dd = self.head_pd_vs_dd(fused_features)
        
        return logits_hc_vs_pd, logits_pd_vs_dd