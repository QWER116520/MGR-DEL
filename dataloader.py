import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
import jieba
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from PIL import Image
from transformers import BlipProcessor


label_dict = {
    "real": 0,
    "fake": 1,
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
}

label_dict_ftr_pred = {
    "real": 0,
    "fake": 1,
    "other": 2,
    0: 0,
    1: 1,
    2: 2,
    "0": 0,
    "1": 1,
    "2": 2,
}

def word2input(texts, max_len, tokenizer):
    token_ids = []
    for i, text in enumerate(texts):
        # 如果是 None 或 NaN，就当成空字符串，避免 tokenizer 报错
        if text is None or (isinstance(text, float) and np.isnan(text)):
            text = ''
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def get_dataloader(path, max_len, batch_size, shuffle, bert_path, data_type, language, blip_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    blip_processor = BlipProcessor.from_pretrained(blip_path)

    # 定义category映射字典
    category_dict = {
        "经济": 0, "健康": 1, "军事": 2, "科学": 3, "政治": 4,
        "国际": 5, "教育": 6, "娱乐": 7, "社会": 8
    }

    if data_type == 'rationale':
        data_list = json.load(open(path, 'r',encoding='utf-8'))
        df_data = pd.DataFrame(columns=('content','label'))
        for item in data_list:
            tmp_data = {}

            # content info
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item['source_id']

            tmp_data['FTR_2'] = item['td_rationale']
            tmp_data['FTR_3'] = item['cs_rationale']

            tmp_data['FTR_2_pred'] = item['td_pred']
            tmp_data['FTR_3_pred'] = item['cs_pred']

            tmp_data['FTR_2_acc'] = item['td_acc']
            tmp_data['FTR_3_acc'] = item['cs_acc']

            tmp_data['image_description'] = item['image_description']
            tmp_data['image_path'] = item['image_path']
            
            # 添加category字段处理
            tmp_data['category'] = category_dict.get(item.get('category', '社会'), 8)  # 默认为社会类别(8)

            df_data = pd.concat([df_data, pd.DataFrame([tmp_data])], ignore_index=True)

        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        id = torch.tensor(df_data['id'].to_numpy())
        category = torch.tensor(df_data['category'].astype(int).to_numpy())  # 新增category处理

        FTR_2_pred = torch.tensor(df_data['FTR_2_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())
        FTR_3_pred = torch.tensor(df_data['FTR_3_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())

        FTR_2_acc = torch.tensor(df_data['FTR_2_acc'].astype(int).to_numpy())
        FTR_3_acc = torch.tensor(df_data['FTR_3_acc'].astype(int).to_numpy())

        FTR_2 = df_data['FTR_2'].to_numpy()
        FTR_3 = df_data['FTR_3'].to_numpy()

        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        
        FTR_2_token_ids, FTR_2_masks = word2input(FTR_2, max_len, tokenizer)
        FTR_3_token_ids, FTR_3_masks = word2input(FTR_3, max_len, tokenizer)

        RP = df_data['image_description'].to_numpy()
        RP_token_ids, RP_masks = word2input(RP, max_len, tokenizer)

        img_paths = df_data['image_path'].to_numpy()
        imgs = []
        for p in img_paths:
            try:
                imgs.append(Image.open(p).convert('RGB'))
            except Exception:
                # 若图像缺失或路径无效，使用占位白图避免中断
                imgs.append(Image.new('RGB', (224, 224), color='white'))
        img_inputs = blip_processor(images=imgs, return_tensors='pt')
        pixel_values = img_inputs['pixel_values']  # Tensor shape: (N, 3, H, W)

        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                FTR_2_pred,
                                FTR_2_acc,
                                FTR_3_pred,
                                FTR_3_acc,
                                FTR_2_token_ids,
                                FTR_2_masks,
                                FTR_3_token_ids,
                                FTR_3_masks,
                                RP_token_ids,        # ✅ 新增
                                RP_masks,            # ✅ 新增
                                pixel_values,        # ✅ 新增
                                label,
                                id,
                                category,            # ✅ 新增category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader
    else:
        print('No match data type!')
        exit()