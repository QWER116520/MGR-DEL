import os
import argparse
import json
from utils.utils import get_tensorboard_writer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='ARG')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--language', type=str, default='en')
parser.add_argument('--root_path', type=str)
parser.add_argument('--batchsize', type=int, default=8)   # 极简配置：显存友好的批次大小
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)  # 梯度累积步数
parser.add_argument('--seed', type=int, default=3759)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--co_attention_dim', type=int, default=300)  # 固定为经验最优值
parser.add_argument('--lr', type=float, default=1e-5)  # 极简配置：BERT推荐学习率
parser.add_argument('--save_log_dir', type=str, default= './logs')
parser.add_argument('--save_param_dir', type=str, default= './param_model')
parser.add_argument('--param_log_dir', type=str, default = './logs/param')

# extra parameter
parser.add_argument('--tensorboard_dir', type=str, default='./logs/tensorlog')
parser.add_argument('--bert_path', type=str, default = './Chi-bert-train')
parser.add_argument('--data_type', type=str, default = 'rationale')
parser.add_argument('--data_name', type=str)
parser.add_argument('--eval_mode', type=bool, default = False)

# model structure control
parser.add_argument('--expert_interaction_method', type=str, default = 'cross_attention')
parser.add_argument('--llm_judgment_predictor_weight', type=float, default = -1)
parser.add_argument('--rationale_usefulness_evaluator_weight', type=float, default = -1)

# DPLE configuration
parser.add_argument('--num_shared_expert', type=int, default=4, help='Number of shared experts')
parser.add_argument('--use_shared_experts', type=bool, default=True, help='Whether to use shared experts')
parser.add_argument('--use_enhanced_gating', type=bool, default=True, help='Whether to use enhanced gating')
parser.add_argument('--use_sequence_experts', type=bool, default=True, help='Whether to use sequence-level experts')
parser.add_argument('--use_multimodal_fusion', type=bool, default=True, help='Whether to use multimodal expert fusion')
parser.add_argument('--load_balance_weight', type=float, default=0.01, help='Load balance loss weight')

# distill config
parser.add_argument('--kd_loss_weight', type=float, default=1)
parser.add_argument('--teacher_path', type=str)
parser.add_argument('--blip_path', type=str, default='./blip-image-captioning-base')


parser.add_argument('--alpha_text', type=float, default=1.0)   # 文本主干权重（固定为均衡值）
parser.add_argument('--alpha_t',    type=float, default=1.0)   # 文本描述理由权重（固定为均衡值）
parser.add_argument('--alpha_c',    type=float, default=1.0)   # 常识理由权重（固定为均衡值）
parser.add_argument('--alpha_rp',   type=float, default=1.0)   # 视觉常识推理权重（固定为均衡值）
parser.add_argument('--alpha_img',  type=float, default=1.0)   # 图像主干权重（固定为均衡值）
parser.add_argument('--dropout',    type=float, default=0.1)   # MLP dropout（固定为标准值）
parser.add_argument('--weight_decay', type=float, default=1e-4) # 优化器权重衰减（固定为标准值）

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from grid_search import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {};'.format \
    (args.lr, args.model_name, args.batchsize, args.epoch, args.gpu))
print('data_type: {}; data_path: {}; data_name: {};'.format \
    (args.data_type, args.root_path, args.data_name))

config = {
        'use_cuda': True,
        'seed': args.seed,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'language': args.language,
        'root_path': args.root_path,
        'weight_decay': args.weight_decay,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': args.dropout},
            'llm_judgment_predictor_weight': args.llm_judgment_predictor_weight,
            'rationale_usefulness_evaluator_weight': args.rationale_usefulness_evaluator_weight,
            'kd_loss_weight': args.kd_loss_weight,
            'alpha_text': args.alpha_text,
            'alpha_t':    args.alpha_t,
            'alpha_c':    args.alpha_c,
            'alpha_rp':   args.alpha_rp,
            'alpha_img':  args.alpha_img
            },
        'emb_dim': args.emb_dim,
        'co_attention_dim': args.co_attention_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir,

        'tensorboard_dir': args.tensorboard_dir,
        'bert_path': args.bert_path,
        'data_type': args.data_type,
        'data_name': args.data_name,
        'eval_mode': args.eval_mode,
        'dropout': args.dropout,

        'teacher_path': args.teacher_path,
        'blip_path': args.blip_path,
        'month': 1,
        
        # DPLE configuration
        'num_shared_expert': args.num_shared_expert,
        'use_shared_experts': args.use_shared_experts,
        'use_enhanced_gating': args.use_enhanced_gating,
        'use_sequence_experts': args.use_sequence_experts,
        'use_multimodal_fusion': args.use_multimodal_fusion,
        'load_balance_weight': args.load_balance_weight
        }

if __name__ == '__main__':
    writer = get_tensorboard_writer(config)
    print('before in config')
    print(config)
    best_metric = Run(config = config, writer = writer).main()

    save_dir = './logs/log'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, config['data_name']+'.json')
    with open(save_path, 'w') as file:
        json.dump(best_metric, file, indent=4, ensure_ascii=False)
