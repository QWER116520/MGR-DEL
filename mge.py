import torch
import torch.nn as nn
from .layers import *
from sklearn.metrics import *
from transformers import BertModel, BlipForConditionalGeneration
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader
from utils.utils import get_monthly_path, get_tensorboard_writer
import json
import numpy as np
from datetime import datetime


class ErrorAnalyzer:
    def __init__(self, save_dir="error_analysis", confidence_threshold=0.5, error_txt_path="error.txt"):
        self.save_dir = save_dir
        self.confidence_threshold = confidence_threshold
        self.error_samples = []
        self.correct_samples = []
        self.error_txt_path = error_txt_path
        self.current_epoch_errors = []
        self.best_model_errors = []
        self.error_txt_initialized = False
        os.makedirs(save_dir, exist_ok=True)
        self.stats = {
            'total_samples': 0,
            'correct_predictions': 0,
            'wrong_predictions': 0,
            'high_confidence_errors': 0,
            'low_confidence_correct': 0,
        }

    def _init_error_txt(self):
        if self.error_txt_initialized:
            return
        with open(self.error_txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("error record\n")
            f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"threshold: {self.confidence_threshold}\n")
            f.write(f"error count: {len(self.best_model_errors)}\n")
            f.write("=" * 80 + "\n\n")
        self.error_txt_initialized = True

    def _write_best_errors_to_txt(self):
        self._init_error_txt()
        with open(self.error_txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("error record\n")
            f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"threshold: {self.confidence_threshold}\n")
            f.write(f"error count: {len(self.best_model_errors)}\n")
            f.write("=" * 80 + "\n\n")
            for idx, error_info in enumerate(self.best_model_errors, 1):
                f.write(f"error #{idx}\n")
                f.write(f"time: {error_info['timestamp']}\n")
                f.write(f"pred prob: {error_info['prediction_prob']:.4f}\n")
                f.write(f"true label: {error_info['true_label']}\n")
                f.write(f"confidence: {error_info['confidence']:.4f}\n")
                if 'step' in error_info and error_info['step'] is not None:
                    f.write(f"step: {error_info['step']}\n")
                if 'category' in error_info:
                    f.write(f"category: {error_info['category']}\n")
                if 'content_preview' in error_info:
                    f.write(f"content:\n{error_info['content_preview']}\n")
                if error_info['confidence'] > 0.7:
                    f.write("high confidence error\n")
                f.write("-" * 60 + "\n\n")

    def on_best_model_saved(self):
        if not self.current_epoch_errors:
            return
        self.best_model_errors = self.current_epoch_errors.copy()
        self._write_best_errors_to_txt()
        self.current_epoch_errors = []

    def on_epoch_end(self):
        if self.current_epoch_errors:
            self.current_epoch_errors = []

    def analyze_batch(self, predictions, labels, batch_data=None, step=None):
        batch_size = len(predictions)
        pred_binary = (predictions > self.confidence_threshold).int()
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        pred_probs = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else predictions

        for i in range(batch_size):
            sample_info = {
                'step': step,
                'prediction_prob': float(pred_probs[i]),
                'prediction_binary': int(pred_binary[i]),
                'true_label': int(labels_np[i]),
                'confidence': abs(pred_probs[i] - 0.5) * 2,
                'timestamp': datetime.now().isoformat()
            }
            if batch_data is not None:
                if 'content' in batch_data and i < len(batch_data['content']):
                    sample_info['content_preview'] = str(batch_data['content'][i])[:200] + "..."
                if 'category' in batch_data and i < len(batch_data['category']):
                    sample_info['category'] = int(batch_data['category'][i])
            is_correct = (pred_binary[i] == labels_np[i])
            if is_correct:
                self.stats['correct_predictions'] += 1
                if sample_info['confidence'] < 0.3:
                    self.stats['low_confidence_correct'] += 1
                    self.correct_samples.append(sample_info)
            else:
                self.stats['wrong_predictions'] += 1
                if sample_info['confidence'] > 0.7:
                    self.stats['high_confidence_errors'] += 1
                self.error_samples.append(sample_info)
                self.current_epoch_errors.append(sample_info)
            self.stats['total_samples'] += 1

    def get_current_stats(self):
        if self.stats['total_samples'] == 0:
            return "no data"
        accuracy = self.stats['correct_predictions'] / self.stats['total_samples']
        error_rate = self.stats['wrong_predictions'] / self.stats['total_samples']
        return {
            'accuracy': f"{accuracy:.4f}",
            'error_rate': f"{error_rate:.4f}",
            'total_samples': self.stats['total_samples'],
            'wrong_predictions': self.stats['wrong_predictions'],
            'high_confidence_errors': self.stats['high_confidence_errors'],
            'low_confidence_correct': self.stats['low_confidence_correct'],
            'current_epoch_errors': len(self.current_epoch_errors),
            'best_model_errors': len(self.best_model_errors)
        }

    def save_error_analysis(self, epoch=None, phase="train"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_analysis_{phase}_{timestamp}"
        if epoch is not None:
            filename += f"_epoch{epoch}"
        error_file = os.path.join(self.save_dir, f"{filename}_errors.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.stats,
                'error_samples': self.error_samples[-100:],
                'metadata': {
                    'phase': phase,
                    'epoch': epoch,
                    'timestamp': timestamp,
                    'confidence_threshold': self.confidence_threshold
                }
            }, f, ensure_ascii=False, indent=2)
        stats_file = os.path.join(self.save_dir, f"{filename}_stats.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"error report ({phase.upper()})\n")
            f.write(f"time: {timestamp}\n")
            if epoch is not None:
                f.write(f"epoch: {epoch}\n")
            f.write(f"threshold: {self.confidence_threshold}\n\n")
            stats = self.get_current_stats()
            f.write(f"total: {stats['total_samples']}\n")
            f.write(f"acc: {stats['accuracy']}\n")
            f.write(f"error rate: {stats['error_rate']}\n")
            f.write(f"errors: {stats['wrong_predictions']}\n")
            f.write(f"high conf errors: {stats['high_confidence_errors']}\n")
            f.write(f"low conf correct: {stats['low_confidence_correct']}\n")
        return error_file, stats_file

    def print_realtime_stats(self, step=None):
        if self.stats['total_samples'] == 0:
            return
        stats = self.get_current_stats()
        prefix = f"[Step {step}] " if step is not None else ""
        print(f"\nstats: {prefix}")
        print(f"  acc: {stats['accuracy']} | errors: {stats['wrong_predictions']}")
        print(f"  high conf errors: {stats['high_confidence_errors']} | low conf correct: {stats['low_confidence_correct']}")
        if len(self.error_samples) > 0:
            latest_error = self.error_samples[-1]
            print(f"  latest error: pred={latest_error['prediction_prob']:.3f}, true={latest_error['true_label']}, conf={latest_error['confidence']:.3f}")


class MGE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.domain_num = 9
        self.num_expert = 6
        self.num_shared_expert = config.get('num_shared_expert', 4)
        self.emb_dim = config['emb_dim']
        self.config = config
        self.use_shared_experts = config.get('use_shared_experts', True)
        self.use_enhanced_gating = config.get('use_enhanced_gating', True)
        self.use_sequence_experts = config.get('use_sequence_experts', True)
        self.use_multimodal_fusion = config.get('use_multimodal_fusion', True)
        self.load_balance_weight = config.get('load_balance_weight', 0.01)

        self.blip_model = BlipForConditionalGeneration.from_pretrained(config['blip_path'], local_files_only=True).vision_model
        for param in self.blip_model.parameters():
            param.requires_grad = False

        self.bert_content = BertModel.from_pretrained(config['bert_path'], local_files_only=True, ignore_mismatched_sizes=True).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(config['bert_path'], local_files_only=True, ignore_mismatched_sizes=True).requires_grad_(False)

        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.bert_FTR.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=self.emb_dim)

        if self.use_shared_experts:
            self.content_shared_experts = SharedExpertNetwork(
                input_dim=self.emb_dim, expert_dim=self.emb_dim, num_experts=self.num_shared_expert
            )
            self.ftr_shared_experts = SharedExpertNetwork(
                input_dim=self.emb_dim, expert_dim=self.emb_dim, num_experts=self.num_shared_expert
            )

        self.load_balance_loss_fn = LoadBalanceLoss(alpha=self.load_balance_weight)

        if self.use_sequence_experts:
            total_experts = self.num_expert + (self.num_shared_expert if self.use_shared_experts else 0)
            self.content_sequence_processor = SequenceLevelExpertProcessor(
                emb_dim=self.emb_dim, num_experts=total_experts, max_seq_len=config.get('max_len', 170)
            )
            self.ftr_2_sequence_processor = SequenceLevelExpertProcessor(
                emb_dim=self.emb_dim, num_experts=total_experts, max_seq_len=config.get('max_len', 170)
            )
            self.ftr_3_sequence_processor = SequenceLevelExpertProcessor(
                emb_dim=self.emb_dim, num_experts=total_experts, max_seq_len=config.get('max_len', 170)
            )

        if self.use_multimodal_fusion:
            self.multimodal_fusion = MultiModalExpertFusion(
                text_dim=self.emb_dim, image_dim=self.emb_dim, output_dim=self.emb_dim, num_fusion_experts=4
            )

        content_expert_list = []
        for i in range(self.domain_num):
            content_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.emb_dim, self.emb_dim)
                )
                content_expert.append(expert)
            content_expert = nn.ModuleList(content_expert)
            content_expert_list.append(content_expert)
        self.content_experts = nn.ModuleList(content_expert_list)

        ftr_expert_list = []
        for i in range(self.domain_num):
            ftr_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.emb_dim, self.emb_dim)
                )
                ftr_expert.append(expert)
            ftr_expert = nn.ModuleList(ftr_expert)
            ftr_expert_list.append(ftr_expert)
        self.ftr_experts = nn.ModuleList(ftr_expert_list)

        content_gate_list = []
        for i in range(self.domain_num):
            if self.use_enhanced_gating:
                total_experts = self.num_expert + (self.num_shared_expert if self.use_shared_experts else 0)
                content_gate = EnhancedGatingNetwork(
                    input_dim=self.emb_dim * 2, num_experts=total_experts, use_temperature=True
                )
            else:
                content_gate = nn.Sequential(
                    nn.Linear(self.emb_dim * 2, self.emb_dim),
                    nn.SiLU(),
                    nn.Linear(self.emb_dim, self.num_expert),
                    nn.Dropout(0.1),
                    nn.Softmax(dim=1)
                )
            content_gate_list.append(content_gate)
        self.content_gate_list = nn.ModuleList(content_gate_list)

        ftr_gate_list = []
        for i in range(self.domain_num):
            if self.use_enhanced_gating:
                total_experts = self.num_expert + (self.num_shared_expert if self.use_shared_experts else 0)
                ftr_gate = EnhancedGatingNetwork(
                    input_dim=self.emb_dim * 2, num_experts=total_experts, use_temperature=True
                )
            else:
                ftr_gate = nn.Sequential(
                    nn.Linear(self.emb_dim * 2, self.emb_dim),
                    nn.SiLU(),
                    nn.Linear(self.emb_dim, self.num_expert),
                    nn.Dropout(0.1),
                    nn.Softmax(dim=1)
                )
            ftr_gate_list.append(ftr_gate)
        self.ftr_gate_list = nn.ModuleList(ftr_gate_list)

        if self.use_shared_experts:
            self.content_adaptive_fusion = AdaptiveFusion(feature_dim=self.emb_dim)
            self.ftr_adaptive_fusion = AdaptiveFusion(feature_dim=self.emb_dim)

        self.aggregator = MaskAttention(config['emb_dim'])
        self.mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'])

        self.hard_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_2 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Linear(config['model']['mlp']['dims'][-1], 1),
            nn.Sigmoid()
        )
        self.score_mapper_ftr_2 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['model']['mlp']['dims'][-1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.hard_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_3 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Linear(config['model']['mlp']['dims'][-1], 1),
            nn.Sigmoid()
        )
        self.score_mapper_ftr_3 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['model']['mlp']['dims'][-1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.simple_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_2 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Linear(config['model']['mlp']['dims'][-1], 3)
        )
        self.simple_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_3 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Linear(config['model']['mlp']['dims'][-1], 3)
        )

        self.content_attention = MaskAttention(config['emb_dim'])
        self.co_attention_2 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        FTR_2, FTR_2_masks = kwargs['FTR_2'], kwargs['FTR_2_masks']
        FTR_3, FTR_3_masks = kwargs['FTR_3'], kwargs['FTR_3_masks']

        category = kwargs.get('category', [8] * content.size(0))
        if isinstance(category, torch.Tensor):
            category = category.tolist()

        pixel_values = kwargs['pixel_values']
        vision_outputs = self.blip_model(pixel_values=pixel_values)
        image_feature = vision_outputs.last_hidden_state.mean(dim=1)

        content_feature = self.bert_content(content, attention_mask=content_masks)[0]
        FTR_2_feature = self.bert_FTR(FTR_2, attention_mask=FTR_2_masks)[0]
        FTR_3_feature = self.bert_FTR(FTR_3, attention_mask=FTR_3_masks)[0]

        batch_size = content.size(0)
        device = content.device
        category_tensor = torch.tensor(category, dtype=torch.long, device=device)
        domain_embeddings = self.domain_embedder(category_tensor)

        content_pooled = content_feature.mean(dim=1)
        content_gate_input = torch.cat([domain_embeddings, content_pooled], dim=-1)

        enhanced_content_features = []
        all_content_gate_weights = []

        for i in range(batch_size):
            domain_idx = category[i]
            if self.use_enhanced_gating:
                gate_weights = self.content_gate_list[domain_idx](content_gate_input[i:i+1], training=self.training)
            else:
                gate_weights = self.content_gate_list[domain_idx](content_gate_input[i:i+1])
            all_content_gate_weights.append(gate_weights)

            domain_expert_outputs = []
            for j in range(self.num_expert):
                expert_output = self.content_experts[domain_idx][j](content_pooled[i:i+1])
                domain_expert_outputs.append(expert_output)
            domain_expert_outputs = torch.stack(domain_expert_outputs, dim=1)

            if self.use_shared_experts:
                shared_expert_outputs = self.content_shared_experts(content_pooled[i:i+1])
                shared_expert_outputs = torch.stack(shared_expert_outputs, dim=1)
                all_expert_outputs = torch.cat([domain_expert_outputs, shared_expert_outputs], dim=1)
            else:
                all_expert_outputs = domain_expert_outputs

            weighted_output = torch.sum(all_expert_outputs * gate_weights.unsqueeze(-1), dim=1)

            if self.use_shared_experts:
                domain_weights = gate_weights[:, :self.num_expert]
                shared_weights = gate_weights[:, self.num_expert:]
                domain_output = torch.sum(domain_expert_outputs * domain_weights.unsqueeze(-1), dim=1)
                shared_output = torch.sum(shared_expert_outputs * shared_weights.unsqueeze(-1), dim=1)
                fused_output = self.content_adaptive_fusion(content_pooled[i:i+1], domain_output, shared_output)
                enhanced_content_features.append(fused_output)
            else:
                enhanced_content_features.append(weighted_output)

        enhanced_content_pooled = torch.cat(enhanced_content_features, dim=0)
        all_content_gate_weights = torch.cat(all_content_gate_weights, dim=0)

        FTR_2_pooled = FTR_2_feature.mean(dim=1)
        FTR_3_pooled = FTR_3_feature.mean(dim=1)
        ftr_2_gate_input = torch.cat([domain_embeddings, FTR_2_pooled], dim=-1)
        ftr_3_gate_input = torch.cat([domain_embeddings, FTR_3_pooled], dim=-1)

        enhanced_ftr_2_features = []
        enhanced_ftr_3_features = []
        all_ftr_2_gate_weights = []
        all_ftr_3_gate_weights = []

        for i in range(batch_size):
            domain_idx = category[i]

            if self.use_enhanced_gating:
                ftr_2_gate_weights = self.ftr_gate_list[domain_idx](ftr_2_gate_input[i:i+1], training=self.training)
            else:
                ftr_2_gate_weights = self.ftr_gate_list[domain_idx](ftr_2_gate_input[i:i+1])
            all_ftr_2_gate_weights.append(ftr_2_gate_weights)

            ftr_2_domain_outputs = []
            for j in range(self.num_expert):
                expert_output = self.ftr_experts[domain_idx][j](FTR_2_pooled[i:i+1])
                ftr_2_domain_outputs.append(expert_output)
            ftr_2_domain_outputs = torch.stack(ftr_2_domain_outputs, dim=1)

            if self.use_shared_experts:
                ftr_2_shared_outputs = self.ftr_shared_experts(FTR_2_pooled[i:i+1])
                ftr_2_shared_outputs = torch.stack(ftr_2_shared_outputs, dim=1)
                ftr_2_all_outputs = torch.cat([ftr_2_domain_outputs, ftr_2_shared_outputs], dim=1)
            else:
                ftr_2_all_outputs = ftr_2_domain_outputs

            weighted_ftr_2 = torch.sum(ftr_2_all_outputs * ftr_2_gate_weights.unsqueeze(-1), dim=1)

            if self.use_shared_experts:
                ftr_2_domain_weights = ftr_2_gate_weights[:, :self.num_expert]
                ftr_2_shared_weights = ftr_2_gate_weights[:, self.num_expert:]
                ftr_2_domain_output = torch.sum(ftr_2_domain_outputs * ftr_2_domain_weights.unsqueeze(-1), dim=1)
                ftr_2_shared_output = torch.sum(ftr_2_shared_outputs * ftr_2_shared_weights.unsqueeze(-1), dim=1)
                fused_ftr_2 = self.ftr_adaptive_fusion(FTR_2_pooled[i:i+1], ftr_2_domain_output, ftr_2_shared_output)
                enhanced_ftr_2_features.append(fused_ftr_2)
            else:
                enhanced_ftr_2_features.append(weighted_ftr_2)

            if self.use_enhanced_gating:
                ftr_3_gate_weights = self.ftr_gate_list[domain_idx](ftr_3_gate_input[i:i+1], training=self.training)
            else:
                ftr_3_gate_weights = self.ftr_gate_list[domain_idx](ftr_3_gate_input[i:i+1])
            all_ftr_3_gate_weights.append(ftr_3_gate_weights)

            ftr_3_domain_outputs = []
            for j in range(self.num_expert):
                expert_output = self.ftr_experts[domain_idx][j](FTR_3_pooled[i:i+1])
                ftr_3_domain_outputs.append(expert_output)
            ftr_3_domain_outputs = torch.stack(ftr_3_domain_outputs, dim=1)

            if self.use_shared_experts:
                ftr_3_shared_outputs = self.ftr_shared_experts(FTR_3_pooled[i:i+1])
                ftr_3_shared_outputs = torch.stack(ftr_3_shared_outputs, dim=1)
                ftr_3_all_outputs = torch.cat([ftr_3_domain_outputs, ftr_3_shared_outputs], dim=1)
            else:
                ftr_3_all_outputs = ftr_3_domain_outputs

            weighted_ftr_3 = torch.sum(ftr_3_all_outputs * ftr_3_gate_weights.unsqueeze(-1), dim=1)

            if self.use_shared_experts:
                ftr_3_domain_weights = ftr_3_gate_weights[:, :self.num_expert]
                ftr_3_shared_weights = ftr_3_gate_weights[:, self.num_expert:]
                ftr_3_domain_output = torch.sum(ftr_3_domain_outputs * ftr_3_domain_weights.unsqueeze(-1), dim=1)
                ftr_3_shared_output = torch.sum(ftr_3_shared_outputs * ftr_3_shared_weights.unsqueeze(-1), dim=1)
                fused_ftr_3 = self.ftr_adaptive_fusion(FTR_3_pooled[i:i+1], ftr_3_domain_output, ftr_3_shared_output)
                enhanced_ftr_3_features.append(fused_ftr_3)
            else:
                enhanced_ftr_3_features.append(weighted_ftr_3)

        enhanced_ftr_2_pooled = torch.cat(enhanced_ftr_2_features, dim=0)
        enhanced_ftr_3_pooled = torch.cat(enhanced_ftr_3_features, dim=0)
        all_ftr_2_gate_weights = torch.cat(all_ftr_2_gate_weights, dim=0)
        all_ftr_3_gate_weights = torch.cat(all_ftr_3_gate_weights, dim=0)

        if self.use_sequence_experts:
            content_feature_enhanced = self.content_sequence_processor(
                sequence_features=content_feature, expert_weights=all_content_gate_weights, mask=content_masks
            )
            FTR_2_feature_enhanced = self.ftr_2_sequence_processor(
                sequence_features=FTR_2_feature, expert_weights=all_ftr_2_gate_weights, mask=FTR_2_masks
            )
            FTR_3_feature_enhanced = self.ftr_3_sequence_processor(
                sequence_features=FTR_3_feature, expert_weights=all_ftr_3_gate_weights, mask=FTR_3_masks
            )
        else:
            content_feature_enhanced = enhanced_content_pooled.unsqueeze(1).expand(-1, content_feature.size(1), -1)
            FTR_2_feature_enhanced = enhanced_ftr_2_pooled.unsqueeze(1).expand(-1, FTR_2_feature.size(1), -1)
            FTR_3_feature_enhanced = enhanced_ftr_3_pooled.unsqueeze(1).expand(-1, FTR_3_feature.size(1), -1)

        content_feature_1, content_feature_2 = content_feature_enhanced, content_feature_enhanced

        mutual_content_FTR_2, _ = self.cross_attention_content_2(content_feature_2, FTR_2_feature_enhanced, content_masks)
        expert_2 = torch.mean(mutual_content_FTR_2, dim=1)

        mutual_content_FTR_3, _ = self.cross_attention_content_3(content_feature_2, FTR_3_feature_enhanced, content_masks)
        expert_3 = torch.mean(mutual_content_FTR_3, dim=1)

        mutual_FTR_content_2, _ = self.cross_attention_ftr_2(FTR_2_feature_enhanced, content_feature_2, FTR_2_masks)
        mutual_FTR_content_2 = torch.mean(mutual_FTR_content_2, dim=1)

        mutual_FTR_content_3, _ = self.cross_attention_ftr_3(FTR_3_feature_enhanced, content_feature_2, FTR_3_masks)
        mutual_FTR_content_3 = torch.mean(mutual_FTR_content_3, dim=1)

        hard_ftr_2_pred = self.hard_mlp_ftr_2(mutual_FTR_content_2).squeeze(1)
        hard_ftr_3_pred = self.hard_mlp_ftr_3(mutual_FTR_content_3).squeeze(1)

        simple_ftr_2_pred = self.simple_mlp_ftr_2(self.simple_ftr_2_attention(FTR_2_feature_enhanced)[0]).squeeze(1)
        simple_ftr_3_pred = self.simple_mlp_ftr_3(self.simple_ftr_3_attention(FTR_3_feature_enhanced)[0]).squeeze(1)

        attn_content, _ = self.content_attention(content_feature_1, mask=content_masks)

        reweight_score_ftr_2 = self.score_mapper_ftr_2(mutual_FTR_content_2)
        reweight_score_ftr_3 = self.score_mapper_ftr_3(mutual_FTR_content_3)

        reweight_expert_2 = reweight_score_ftr_2 * expert_2
        reweight_expert_3 = reweight_score_ftr_3 * expert_3

        if self.use_multimodal_fusion:
            multimodal_fused = self.multimodal_fusion(
                text_features=enhanced_content_pooled, image_features=image_feature
            )
            all_feature = torch.cat(
                (attn_content.unsqueeze(1), reweight_expert_2.unsqueeze(1), reweight_expert_3.unsqueeze(1), multimodal_fused.unsqueeze(1)),
                dim=1
            )
        else:
            all_feature = torch.cat(
                (attn_content.unsqueeze(1), reweight_expert_2.unsqueeze(1), reweight_expert_3.unsqueeze(1), image_feature.unsqueeze(1)),
                dim=1
            )

        final_feature, _ = self.aggregator(all_feature)
        label_pred = self.mlp(final_feature)

        gate_value = torch.concat([reweight_score_ftr_2, reweight_score_ftr_3], dim=1)

        all_gate_weights = torch.cat([all_content_gate_weights, all_ftr_2_gate_weights, all_ftr_3_gate_weights], dim=0)
        load_balance_loss = self.load_balance_loss_fn(all_gate_weights)

        res = {
            'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
            'gate_value': gate_value,
            'final_feature': final_feature,
            'content_feature': attn_content,
            'ftr_2_feature': reweight_expert_2,
            'ftr_3_feature': reweight_expert_3,
            'load_balance_loss': load_balance_loss,
            'enhanced_content_pooled': enhanced_content_pooled,
            'enhanced_ftr_2_pooled': enhanced_ftr_2_pooled,
            'enhanced_ftr_3_pooled': enhanced_ftr_3_pooled,
            'content_gate_weights': all_content_gate_weights,
            'ftr_2_gate_weights': all_ftr_2_gate_weights,
            'ftr_3_gate_weights': all_ftr_3_gate_weights,
        }

        if self.use_multimodal_fusion:
            res['multimodal_fused'] = multimodal_fused

        res['hard_ftr_2_pred'] = hard_ftr_2_pred
        res['hard_ftr_3_pred'] = hard_ftr_3_pred
        res['simple_ftr_2_pred'] = simple_ftr_2_pred
        res['simple_ftr_3_pred'] = simple_ftr_3_pred

        return res


class Trainer:
    def __init__(self, config, writer):
        self.config = config
        self.writer = writer
        self.num_expert = 2

        error_analysis_dir = os.path.join(config.get('save_param_dir', './'), 'error_analysis')
        error_txt_path = config.get('error_txt_path', 'error.txt')
        self.error_analyzer = ErrorAnalyzer(
            save_dir=error_analysis_dir,
            confidence_threshold=config.get('error_analysis_threshold', 0.5),
            error_txt_path=error_txt_path
        )

        self.save_path = os.path.join(
            self.config['save_param_dir'],
            self.config['model_name'] + '_' + self.config['data_name'],
            str(self.config['month'])
        )
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)

    def train(self, logger=None):
        st_tm = time.time()
        writer = self.writer

        if logger:
            logger.info('start training')
        print('\n\n')
        print('==================== start training ====================')

        self.model = MGE(self.config)

        if self.config['use_cuda']:
            self.model = self.model.cuda()

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)

        recorder = Recorder(self.config['early_stop'])

        train_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'train_classified.json')
        train_loader = get_dataloader(
            train_path, self.config['max_len'], self.config['batchsize'], shuffle=True,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'],
            language=self.config['language'], blip_path=self.config['blip_path']
        )

        val_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'val_classified.json')
        val_loader = get_dataloader(
            val_path, self.config['max_len'], self.config['batchsize'], shuffle=False,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'],
            language=self.config['language'], blip_path=self.config['blip_path']
        )

        test_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'test_classified.json')
        test_future_loader = get_dataloader(
            test_path, self.config['max_len'], self.config['batchsize'], shuffle=False,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'],
            language=self.config['language'], blip_path=self.config['blip_path']
        )

        ed_tm = time.time()
        print('time cost in model and data loading: {}s'.format(ed_tm - st_tm))

        for epoch in range(self.config['epoch']):
            print('---------- epoch {} ----------'.format(epoch))
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss_classify = Averager()
            accumulated_loss_classify = 0.0

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'])
                label = batch_data['label']
                hard_ftr_2_label = batch_data['FTR_2_acc']
                hard_ftr_3_label = batch_data['FTR_3_acc']
                simple_ftr_2_label = batch_data['FTR_2_pred']
                simple_ftr_3_label = batch_data['FTR_3_pred']

                batch_input_data = {**self.config, **batch_data}

                res = self.model(**batch_input_data)
                loss_classify = loss_fn(res['classify_pred'], label.float())

                loss_hard_aux_fn = torch.nn.BCELoss()
                loss_hard_aux = loss_hard_aux_fn(res['hard_ftr_2_pred'], hard_ftr_2_label.float()) + loss_hard_aux_fn(res['hard_ftr_3_pred'], hard_ftr_3_label.float())

                loss_simple_aux_fn = torch.nn.CrossEntropyLoss()
                loss_simple_aux = loss_simple_aux_fn(res['simple_ftr_2_pred'], simple_ftr_2_label.long()) + loss_simple_aux_fn(res['simple_ftr_3_pred'], simple_ftr_3_label.long())

                load_balance_loss = res.get('load_balance_loss', 0)

                loss = loss_classify
                loss += self.config['model']['rationale_usefulness_evaluator_weight'] * loss_hard_aux / self.num_expert
                loss += self.config['model']['llm_judgment_predictor_weight'] * loss_simple_aux / self.num_expert

                if load_balance_loss is not None and load_balance_loss != 0:
                    loss += self.config.get('load_balance_weight', 0.01) * load_balance_loss

                loss = loss / gradient_accumulation_steps
                accumulated_loss_classify += loss_classify.item()

                loss.backward()

                if (step_n + 1) % gradient_accumulation_steps == 0 or (step_n + 1) == len(train_data_iter):
                    optimizer.step()
                    optimizer.zero_grad()
                    avg_loss_classify.add(accumulated_loss_classify)
                    accumulated_loss_classify = 0.0

                self.error_analyzer.analyze_batch(
                    predictions=res['classify_pred'],
                    labels=label,
                    batch_data=batch_data,
                    step=step_n
                )

                if step_n % 100 == 0:
                    self.error_analyzer.print_realtime_stats(step=step_n)

                del res, loss, loss_classify, loss_hard_aux, loss_simple_aux
                del batch_data, batch_input_data, label
                del hard_ftr_2_label, hard_ftr_3_label, simple_ftr_2_label, simple_ftr_3_label
                if 'load_balance_loss' in locals():
                    del load_balance_loss

                if step_n % 50 == 0:
                    torch.cuda.empty_cache()
                if step_n % 200 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            print('----- in val progress... -----')
            results, val_aux_info = self.test(val_loader)
            mark = recorder.add(results)
            print()

            error_files = self.error_analyzer.save_error_analysis(epoch=epoch, phase="train")
            print(f"error analysis saved: {error_files[0]}")

            writer.add_scalar('month_' + str(self.config['month']) + '/train_loss', avg_loss_classify.item(), global_step=epoch)
            writer.add_scalars('month_' + str(self.config['month']) + '/test', results, global_step=epoch)

            if logger:
                logger.info('---------- epoch {} ----------'.format(epoch))
                logger.info('train loss classify: {}'.format(avg_loss_classify.item()))
                logger.info('\n')
                logger.info('val loss classify: {}'.format(val_aux_info['val_avg_loss_classify'].item()))
                logger.info('val result: {}'.format(results))
                logger.info('\n')

            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'parameter_bert.pkl'))
                self.error_analyzer.on_best_model_saved()
            elif mark == 'esc':
                break
            else:
                self.error_analyzer.on_epoch_end()
                continue

        test_dir = os.path.join('./logs/test/', self.config['model_name'] + '_' + self.config['data_name'])
        os.makedirs(test_dir, exist_ok=True)
        test_res_path = os.path.join(test_dir, 'month_' + str(self.config['month']) + '.json')

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bert.pkl')))
        future_results, label, pred, id, ae, acc = self.predict(test_future_loader)

        writer.add_scalars('month_' + str(self.config['month']) + '/test', future_results)

        if logger:
            logger.info("start testing")
            logger.info("test score: {}".format(future_results))
            logger.info("lr: {}, avg test score: {}".format(self.config['lr'], future_results['metric']))
        print('test results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bert.pkl'), epoch

    def test(self, dataloader):
        loss_fn = torch.nn.BCELoss()
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        avg_loss_classify = Averager()

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'])
                batch_label = batch_data['label']
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)

                loss_classify = loss_fn(res['classify_pred'], batch_label.float())

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(res['classify_pred'].detach().cpu().numpy().tolist())
                avg_loss_classify.add(loss_classify.item())

                self.error_analyzer.analyze_batch(
                    predictions=res['classify_pred'],
                    labels=batch_label,
                    batch_data=batch_data,
                    step=step_n
                )

                del res, batch_data, batch_input_data, batch_label, loss_classify
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        aux_info = {'val_avg_loss_classify': avg_loss_classify}
        return metrics(label, pred), aux_info

    def predict(self, dataloader):
        if self.config['eval_mode']:
            self.model = MGE(self.config)
            if self.config['use_cuda']:
                self.model = self.model.cuda()
            print('========== in test process ==========')
            print('now load in test model...')
            self.model.load_state_dict(torch.load(self.config['eval_model_path']))

        pred = []
        label = []
        id = []
        ae = []
        accuracy = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'])
                batch_label = batch_data['label']
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)
                batch_pred = res['classify_pred']

                cur_labels = batch_label.detach().cpu().numpy().tolist()
                cur_preds = batch_pred.detach().cpu().numpy().tolist()
                label.extend(cur_labels)
                pred.extend(cur_preds)
                ae_list = []
                for index in range(len(cur_labels)):
                    ae_list.append(abs(cur_preds[index] - cur_labels[index]))
                accuracy_list = [1 if ae < 0.5 else 0 for ae in ae_list]
                ae.extend(ae_list)
                accuracy.extend(accuracy_list)

        return metrics(label, pred), label, pred, id, ae, accuracy
