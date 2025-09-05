import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertTokenizer, BertModel

class TextTokenizer(nn.Module):
    
    def __init__(self, model_name='bert-base-uncased', output_dim=128, dropout=0.1):
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        input_dim = self.bert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, text_list, device):
        tokens = self.tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        output = outputs.pooler_output
        
        return self.projection(output)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0) 

class MultiheadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.d_k = model_dim // num_heads
        
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, model_dim = query.size()
        seq_len_k = key.size(1)
    
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_w = F.softmax(scores, dim=-1)
        attention_w = self.dropout(attention_w)
        
        context = torch.matmul(attention_w, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.model_dim)
        
        output = self.w_o(context)
        output = self.layer_norm(output + query)  # Residual connection
        
        return output, attention_w


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
        x = self.layer_norm(x + residual)  # Residual connection
        return x


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention_1to2 = MultiheadAttention(model_dim, num_heads, dropout)
        self.cross_attention_2to1 = MultiheadAttention(model_dim, num_heads, dropout)
        self.self_attention_1 = MultiheadAttention(model_dim, num_heads, dropout)
        self.self_attention_2 = MultiheadAttention(model_dim, num_heads, dropout)
        
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)       
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)
        
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
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        seq_len: int = 256,
        num_classes: int = 2,
        use_text: bool = True,  
        text_encoder_dim: int = 128,  
        fusion_method: str = 'concat',
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.use_text = use_text
        self.fusion_method = fusion_method
        
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        
        self.positional_encoding = PositionalEncoding(model_dim, max_len=seq_len)
        
        self.layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # ----------Text encoder ----------
        if use_text:
            self.text_encoder = TextTokenizer(output_dim=text_encoder_dim, dropout=dropout)
            
            if fusion_method == 'concat':
                fusion_dim = model_dim * 2 + text_encoder_dim
            elif fusion_method == 'attention':
                fusion_dim = model_dim * 2
                self.fusion_attention = nn.MultiheadAttention(
                    embed_dim=model_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout
                )
                self.text_to_signal = nn.Linear(text_encoder_dim, model_dim * 2)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
        else:
            fusion_dim = model_dim * 2

        # Classification heads
        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: HC vs PD
        )
        
        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: PD vs DD
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, left_wrist, right_wrist, patient_texts=None, device=None):
        
        left_encoded = self.left_projection(left_wrist)   
        right_encoded = self.right_projection(right_wrist) 

        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        for layer in self.layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)

        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)


        fused_signal_features = torch.cat([left_pool, right_pool], dim=1)  
        
        if self.use_text and patient_texts is not None:
            if device is None:
                device = left_wrist.device
            
            text_features = self.text_encoder(patient_texts, device)
            
            if self.fusion_method == 'concat':
                fused_features = torch.cat([fused_signal_features, text_features], dim=1)
            elif self.fusion_method == 'attention':
                text_transformed = self.text_to_signal(text_features).unsqueeze(1)  
                signal_features = fused_signal_features.unsqueeze(1)  
                
                
                fused_output, _ = self.fusion_attention(
                    query=signal_features,
                    key=text_transformed,
                    value=text_transformed
                )
                fused_features = fused_output.squeeze(1)
        else:
            fused_features = fused_signal_features

        logits_hc_vs_pd = self.head_hc_vs_pd(fused_features)
        logits_pd_vs_dd = self.head_pd_vs_dd(fused_features)

        return logits_hc_vs_pd, logits_pd_vs_dd