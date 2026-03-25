import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Block


class ReverseLayerF(nn.Module):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_size, feature_num, kernel)
            for kernel, feature_num in feature_kernel.items()
        ])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


class MaskAttention(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.attention_layer = nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x), attn


class SelfAttentionFeatureExtract(nn.Module):
    def __init__(self, multi_head_num, input_size, output_size=None):
        super().__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)

    def forward(self, inputs, query, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))
        feature, attn = self.attention(query=query, value=inputs, key=inputs, mask=mask)
        return feature, attn


def masked_softmax(scores, mask):
    scores = scores.masked_fill(mask == 0, -np.inf)
    return F.softmax(scores.float(), dim=-1).type_as(scores)


class ParallelCoAttentionNetwork(nn.Module):
    def __init__(self, hidden_dim, co_attention_dim, mask_in=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.mask_in = mask_in
        self.W_b = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_v = nn.Parameter(torch.randn(co_attention_dim, hidden_dim))
        self.W_q = nn.Parameter(torch.randn(co_attention_dim, hidden_dim))
        self.w_hv = nn.Parameter(torch.randn(co_attention_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(co_attention_dim, 1))

    def forward(self, V, Q, V_mask=None, Q_mask=None):
        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        H_q = nn.Tanh()(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)

        if self.mask_in:
            masked_a_v = masked_softmax(a_v.squeeze(1), V_mask).unsqueeze(1)
            masked_a_q = masked_softmax(a_q.squeeze(1), Q_mask).unsqueeze(1)
            v = torch.squeeze(torch.matmul(masked_a_v, V.permute(0, 2, 1)))
            q = torch.squeeze(torch.matmul(masked_a_q, Q))
            return masked_a_v, masked_a_q, v, q
        else:
            v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
            q = torch.squeeze(torch.matmul(a_q, Q))
            return a_v, a_q, v, q


class SharedExpertNetwork(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, input_dim),
                nn.LayerNorm(input_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        return expert_outputs


class EnhancedGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, use_temperature=True, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.use_temperature = use_temperature
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_experts)
        )
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.ones(1))

    def forward(self, x, training=True):
        logits = self.gate_network(x)
        if training and self.use_temperature:
            gate_weights = F.gumbel_softmax(logits / self.temperature, tau=1.0, hard=False)
        else:
            gate_weights = F.softmax(logits / self.temperature, dim=-1)
        return gate_weights


class AdaptiveFusion(nn.Module):
    def __init__(self, feature_dim, fusion_dim=None, dropout=0.1):
        super().__init__()
        if fusion_dim is None:
            fusion_dim = feature_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(feature_dim * 3, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, original_features, domain_features, shared_features):
        concat_features = torch.cat([original_features, domain_features, shared_features], dim=-1)
        fused_features = self.fusion_network(concat_features)
        output = self.residual_weight * fused_features + (1 - self.residual_weight) * original_features
        return output


class LoadBalanceLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, gate_weights):
        expert_usage = gate_weights.mean(dim=0)
        num_experts = expert_usage.size(0)
        target_usage = 1.0 / num_experts
        load_balance_loss = self.alpha * torch.var(expert_usage)
        return load_balance_loss


class SequenceLevelExpertProcessor(nn.Module):
    def __init__(self, emb_dim, num_experts, max_seq_len=512, num_heads=8):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, emb_dim) * 0.02)
        self.sequence_experts = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 2,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(num_experts)
        ])

    def forward(self, sequence_features, expert_weights, mask=None):
        batch_size, seq_len, emb_dim = sequence_features.shape
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        sequence_features = sequence_features + pos_enc
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        expert_outputs = [expert(sequence_features, src_key_padding_mask=src_key_padding_mask) for expert in self.sequence_experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        expert_weights = expert_weights.unsqueeze(1).unsqueeze(2)
        enhanced_sequence = torch.sum(expert_outputs * expert_weights, dim=-1)
        return enhanced_sequence


class MultiModalExpertFusion(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim, num_fusion_experts=4, dropout=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim
        self.num_fusion_experts = num_fusion_experts
        self.text_projector = nn.Linear(text_dim, output_dim)
        self.image_projector = nn.Linear(image_dim, output_dim)
        self.fusion_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_fusion_experts)
        ])
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, num_fusion_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, text_features, image_features):
        text_aligned = self.text_projector(text_features)
        image_aligned = self.image_projector(image_features)
        concat_features = torch.cat([text_aligned, image_aligned], dim=-1)
        fusion_weights = self.fusion_gate(concat_features)
        fusion_outputs = [expert(concat_features) for expert in self.fusion_experts]
        fusion_outputs = torch.stack(fusion_outputs, dim=1)
        fusion_weights = fusion_weights.unsqueeze(-1)
        fused_features = torch.sum(fusion_outputs * fusion_weights, dim=1)
        return fused_features
