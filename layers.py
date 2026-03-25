import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from timm.models.vision_transformer import Block

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            #layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature

class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)

        return outputs, scores

class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # print('x shape after self attention: {}'.format(x.shape))

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn

class SelfAttentionFeatureExtract(torch.nn.Module):
    def __init__(self, multi_head_num, input_size, output_size=None):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
    def forward(self, inputs, query, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        feature, attn = self.attention(query=query,
                                 value=inputs,
                                 key=inputs,
                                 mask=mask
                                 )
        return feature, attn

def masked_softmax(scores, mask):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""

    # Fill pad positions with -inf
    scores = scores.masked_fill(mask == 0, -np.inf)
 
    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)
 
class ParallelCoAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, co_attention_dim, mask_in=False):
        super(ParallelCoAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.mask_in = mask_in
        # self.src_length_masking = src_length_masking
 
        # [hid_dim, hid_dim]
        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # [co_dim, hid_dim]
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        # [co_dim, hid_dim]
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        # [co_dim, 1]
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        # [co_dim, 1]
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
 
    def forward(self, V, Q, V_mask=None, Q_mask=None):
        """ ori_setting
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        """ new_setting
        :param V: news content, batch_size * hidden_dim * content_length , eg B x 768 x 170
        :param Q: FTR info, batch_size * FTR_length * hidden_dim, eg B x 512 x 768
        :param batch_size: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """

        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        # (batch_size, co_attention_dim, region_num)
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        # (batch_size, co_attention_dim, seq_len)
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
 
        # (batch_size, 1, region_num)
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        # (batch_size, 1, seq_len)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)

        if self.mask_in:
            # # (batch_size, 1, region_num)
            masked_a_v = masked_softmax(
                a_v.squeeze(1), V_mask
            ).unsqueeze(1)
    
            # # (batch_size, 1, seq_len)
            masked_a_q = masked_softmax(
                a_q.squeeze(1), Q_mask
            ).unsqueeze(1)
 
            # (batch_size, hidden_dim)
            v = torch.squeeze(torch.matmul(masked_a_v, V.permute(0, 2, 1)))
            # (batch_size, hidden_dim)
            q = torch.squeeze(torch.matmul(masked_a_q, Q))
    
            return masked_a_v, masked_a_q, v, q
        else:
            # (batch_size, hidden_dim)
            v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
            # (batch_size, hidden_dim)
            q = torch.squeeze(torch.matmul(a_q, Q))
    
            return a_v, a_q, v, q


# ========== DPLE核心组件 ==========

class SharedExpertNetwork(nn.Module):
    """共享专家网络 - DPLE核心组件"""
    
    def __init__(self, input_dim, expert_dim, num_experts, dropout=0.1):
        super(SharedExpertNetwork, self).__init__()
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
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            expert_outputs: list of (batch_size, input_dim)
        """
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        return expert_outputs


class EnhancedGatingNetwork(nn.Module):
    """增强门控网络 - 支持温度参数和Gumbel Softmax"""
    
    def __init__(self, input_dim, num_experts, use_temperature=True, dropout=0.1):
        super(EnhancedGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.use_temperature = use_temperature
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_experts)
        )
        
        # 可学习的温度参数
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.ones(1))
    
    def forward(self, x, training=True):
        """
        Args:
            x: (batch_size, input_dim)
            training: bool, 是否在训练模式
        Returns:
            gate_weights: (batch_size, num_experts)
        """
        logits = self.gate_network(x)
        
        if training and self.use_temperature:
            # 训练时使用Gumbel Softmax增加随机性
            gate_weights = F.gumbel_softmax(logits / self.temperature, tau=1.0, hard=False)
        else:
            # 推理时使用标准softmax
            gate_weights = F.softmax(logits / self.temperature, dim=-1)
        
        return gate_weights


class AdaptiveFusion(nn.Module):
    """自适应融合层 - 融合领域专家和共享专家的输出"""
    
    def __init__(self, feature_dim, fusion_dim=None, dropout=0.1):
        super(AdaptiveFusion, self).__init__()
        if fusion_dim is None:
            fusion_dim = feature_dim
            
        self.fusion_network = nn.Sequential(
            nn.Linear(feature_dim * 3, fusion_dim),  # 原始+领域+共享
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, original_features, domain_features, shared_features):
        """
        Args:
            original_features: (batch_size, feature_dim)
            domain_features: (batch_size, feature_dim)
            shared_features: (batch_size, feature_dim)
        Returns:
            fused_features: (batch_size, feature_dim)
        """
        # 拼接所有特征
        concat_features = torch.cat([original_features, domain_features, shared_features], dim=-1)
        
        # 通过融合网络
        fused_features = self.fusion_network(concat_features)
        
        # 残差连接
        output = self.residual_weight * fused_features + (1 - self.residual_weight) * original_features
        
        return output


class LoadBalanceLoss(nn.Module):
    """负载均衡损失 - 防止专家网络退化"""
    
    def __init__(self, alpha=0.01):
        super(LoadBalanceLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, gate_weights):
        """
        Args:
            gate_weights: (batch_size, num_experts)
        Returns:
            load_balance_loss: scalar
        """
        # 计算每个专家的平均使用率
        expert_usage = gate_weights.mean(dim=0)  # (num_experts,)
        
        # 理想情况下每个专家使用率应该相等
        num_experts = expert_usage.size(0)
        target_usage = 1.0 / num_experts
        
        # 计算使用率方差作为负载均衡损失
        load_balance_loss = self.alpha * torch.var(expert_usage)
        
        return load_balance_loss


class SequenceLevelExpertProcessor(nn.Module):
    """序列级专家处理器 - 避免简单复制，使用Transformer处理序列"""
    
    def __init__(self, emb_dim, num_experts, max_seq_len=512, num_heads=8):
        super(SequenceLevelExpertProcessor, self).__init__()
        self.emb_dim = emb_dim
        self.num_experts = num_experts
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, emb_dim) * 0.02)
        
        # 多个Transformer专家
        self.sequence_experts = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=emb_dim * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_experts)
        ])
        
        # 专家权重融合
        self.expert_fusion = nn.Linear(num_experts, 1)
        
    def forward(self, sequence_features, expert_weights, mask=None):
        """
        Args:
            sequence_features: (batch_size, seq_len, emb_dim)
            expert_weights: (batch_size, num_experts)
            mask: (batch_size, seq_len)
        Returns:
            enhanced_sequence: (batch_size, seq_len, emb_dim)
        """
        batch_size, seq_len, emb_dim = sequence_features.shape
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        sequence_features = sequence_features + pos_enc
        
        # 创建注意力掩码
        if mask is not None:
            # 转换为Transformer格式的掩码
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_mask = attn_mask.expand(-1, -1, seq_len, -1)  # (batch, 1, seq_len, seq_len)
            attn_mask = attn_mask.squeeze(1)  # (batch, seq_len, seq_len)
            # 0表示被掩码，需要转换为True
            src_key_padding_mask = (mask == 0)  # (batch, seq_len)
        else:
            src_key_padding_mask = None
        
        # 多个专家并行处理
        expert_outputs = []
        for expert in self.sequence_experts:
            expert_out = expert(sequence_features, src_key_padding_mask=src_key_padding_mask)
            expert_outputs.append(expert_out)
        
        # 堆叠专家输出
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (batch, seq_len, emb_dim, num_experts)
        
        # 使用专家权重加权融合
        expert_weights = expert_weights.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, num_experts)
        enhanced_sequence = torch.sum(expert_outputs * expert_weights, dim=-1)  # (batch, seq_len, emb_dim)
        
        return enhanced_sequence


class MultiModalExpertFusion(nn.Module):
    """多模态专家融合 - 文本和图像的深度融合"""
    
    def __init__(self, text_dim, image_dim, output_dim, num_fusion_experts=4, dropout=0.1):
        super(MultiModalExpertFusion, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim
        self.num_fusion_experts = num_fusion_experts
        
        # 模态对齐
        self.text_projector = nn.Linear(text_dim, output_dim)
        self.image_projector = nn.Linear(image_dim, output_dim)
        
        # 多个融合专家
        self.fusion_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_fusion_experts)
        ])
        
        # 融合门控网络
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, num_fusion_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: (batch_size, text_dim)
            image_features: (batch_size, image_dim)
        Returns:
            fused_features: (batch_size, output_dim)
        """
        # 模态对齐
        text_aligned = self.text_projector(text_features)
        image_aligned = self.image_projector(image_features)
        
        # 拼接特征用于门控
        concat_features = torch.cat([text_aligned, image_aligned], dim=-1)
        
        # 计算融合权重
        fusion_weights = self.fusion_gate(concat_features)  # (batch, num_fusion_experts)
        
        # 多个融合专家处理
        fusion_outputs = []
        for expert in self.fusion_experts:
            fused = expert(concat_features)
            fusion_outputs.append(fused)
        
        # 堆叠并加权融合
        fusion_outputs = torch.stack(fusion_outputs, dim=1)  # (batch, num_experts, output_dim)
        fusion_weights = fusion_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        
        fused_features = torch.sum(fusion_outputs * fusion_weights, dim=1)  # (batch, output_dim)
        
        return fused_features