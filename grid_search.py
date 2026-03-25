import logging
import os
import json
import random
import torch
import numpy as np

from models.arg import Trainer as ARGTrainer
from models.argd import Trainer as ARGDTrainer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def frange(x, y, jump):
    while x < y:
        x = round(x, 8)
        yield x
        x += jump

class Run():
    def __init__(self,
                 config,
                 writer
                 ):
        self.config = config
        self.writer = writer

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(level = logging.INFO)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] + '_' + self.config['data_name'] +'_'+ 'param.txt')
        logger = self.getFileLogger(param_log_file)

        # 极简版超参数配置 - 只调优最关键的参数
        train_param = {
           # 最关键：学习率（减少到2个选择）
           'lr': [1e-5, 2e-5],
           # 最关键：批次大小（显存友好的选择）
           'batchsize': [8, 16],
       }
       
        # 固定其他参数为经验最优值
        fixed_params = {
            # 损失权重固定为均衡值
            'model.alpha_text': 1.0,
            'model.alpha_t': 1.0, 
            'model.alpha_c': 1.0,
            'model.alpha_rp': 1.0,
            'model.alpha_img': 1.0,
            
            # 正则化固定
            'dropout': 0.1,
            'weight_decay': 1e-4,
            
            # 结构参数固定
            'co_attention_dim': 300,
            'num_shared_expert': 4,
            'load_balance_weight': 0.01,
        }
        
        # 应用固定参数到配置
        for param_key, param_value in fixed_params.items():
            if '.' in param_key:
                keys = param_key.split('.')
                sub = self.config
                for key in keys[:-1]:
                    sub = sub[key]
                sub[keys[-1]] = param_value
            else:
                self.config[param_key] = param_value

        print(train_param)
        param = train_param
        best_param = []

        json_dir = os.path.join(
            './logs/json/',
            self.config['model_name'] + '_' + self.config['data_name']
        )
        json_path = os.path.join(
            json_dir,
            'month_' + str(self.config['month']) + '.json'
        )
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        json_result = []
        # 记录基础保存目录，便于为每组超参数创建独立目录
        base_save_param_dir = self.config['save_param_dir']
        base_param_log_dir = self.config['param_log_dir']
        for p, vs in param.items():
            setup_seed(self.config['seed'])
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                # 根据 key 自动赋值到 config（支持嵌套字段，用 “.” 分层）
                if '.' in p:
                    keys = p.split('.')
                    sub = self.config
                    for key in keys[:-1]:
                        sub = sub[key]
                    sub[keys[-1]] = v
                else:
                    self.config[p] = v

                # 为当前超参数创建独立保存目录（模型与日志）
                run_tag = f"{p.replace('.', '-')}_{str(v).replace('/', '_')}"
                run_save_dir = os.path.join(
                    base_save_param_dir,
                    'experiments',
                    self.config['model_name'] + '_' + self.config['data_name'],
                    f"month_{self.config['month']}",
                    run_tag
                )
                os.makedirs(run_save_dir, exist_ok=True)
                self.config['save_param_dir'] = run_save_dir

                # 为当前超参数创建独立参数日志目录
                run_param_log_dir = os.path.join(
                    base_param_log_dir,
                    'experiments',
                    self.config['model_name'] + '_' + self.config['data_name'],
                    f"month_{self.config['month']}",
                    run_tag
                )
                os.makedirs(run_param_log_dir, exist_ok=True)
                self.config['param_log_dir'] = run_param_log_dir

                if self.config['model_name'] == 'ARG':
                    trainer = ARGTrainer(self.config, self.writer)
                elif self.config['model_name'] == 'ARG-D':
                    trainer = ARGDTrainer(self.config, self.writer)
                else:
                    raise ValueError('model_name is not supported')

                metrics, model_path, train_epochs = trainer.train(logger)
                json_result.append({
                    'lr': self.config['lr'],
                    'metric': metrics,
                    'train_epochs': train_epochs,
                    'param_name': p,
                    'param_value': v,
                    'model_path': model_path,
                    'save_dir': run_save_dir,
                })

                if metrics['metric'] > best_metric['metric']:
                    best_metric = metrics
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best macro f1:", best_metric['metric'])
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('==================================================\n\n')
        # 恢复基础保存目录，避免影响外部调用
        self.config['save_param_dir'] = base_save_param_dir
        self.config['param_log_dir'] = base_param_log_dir
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)

        return best_metric
