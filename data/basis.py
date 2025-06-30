# -*- coding: utf-8 -*-
import os
import json

import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from utils.data_preprocess import get_wrong_ids
import numpy as np


class DataCollator:
    def __init__(self, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [self.tokenizer(t, return_offsets_mapping=True, add_special_tokens=False) for t in ori_texts]
        max_len = max([len(t['input_ids']) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()

        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            off_mapping = encoded_text['offset_mapping']
            for idx in wrong_ids:
                for j, (b, e) in enumerate(off_mapping):
                    if b <= idx < e:
                        # j+1是因为前面的 CLS token
                        det_labels[i, j + 1] = 1
        return list(ori_texts), list(cor_texts), det_labels


class RandomReplDataCollator:
    def __init__(self, tokenizer: BertTokenizerFast):
        from A_generate_confusion_matrix.confusion_t import confusion_matrix
        self.tokenizer = tokenizer
        self.confusion_matrix = confusion_matrix

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [self.tokenizer(t, return_offsets_mapping=True, add_special_tokens=False) for t in ori_texts]
        max_len = max([len(t['input_ids']) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()

        # 初始化替换后的句子列表
        replaced_texts = []

        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            input_ids = encoded_text['input_ids']
            off_mapping = encoded_text['offset_mapping']
            replaced_input_ids = input_ids.copy()  # 用于存储替换后的 token IDs

            for j, (token_id, (b, e)) in enumerate(zip(input_ids, off_mapping)):
                # 检查当前 token 是否在错误位置
                is_wrong = any(b <= idx < e for idx in wrong_ids)
                if not is_wrong:
                    # 以 15% 的概率进行替换
                    if np.random.random() < 0.3:
                        # 获取当前 token 的转移概率分布
                        prob_dist = self.confusion_matrix[token_id, :]
                        if np.sum(prob_dist) < 1e-8:
                            # 如果没有转移概率，跳过替换
                            continue
                        if self.tokenizer.convert_ids_to_tokens(token_id).isdigit():
                            continue
                        # 根据概率分布采样一个 token ID
                        # print(prob_dist)
                        # print(np.sum(prob_dist))
                        sampled_token_id = np.random.choice(len(prob_dist), p=prob_dist)
                        replaced_input_ids[j] = sampled_token_id

            # 将替换后的 token IDs 转换为文本
            replaced_text = self.tokenizer.convert_ids_to_tokens(replaced_input_ids)
            replaced_text = self.tokenizer.convert_tokens_to_string(replaced_text).replace(' ', '')
            replaced_texts.append(replaced_text)

            # 更新 det_labels
            for idx in wrong_ids:
                for j, (b, e) in enumerate(off_mapping):
                    if b <= idx < e:
                        # j+1 是因为前面的 CLS token
                        det_labels[i, j + 1] = 1
                # 校验 cor_texts 和 replaced_texts 的长度一致性
        for i, (cor_text, replaced_text) in enumerate(zip(cor_texts, replaced_texts)):
            cor_tokens = self.tokenizer(cor_text, add_special_tokens=False)['input_ids']
            replaced_tokens = self.tokenizer(replaced_text, add_special_tokens=False)['input_ids']
            if len(cor_tokens) != len(replaced_tokens):
                # 如果长度不一致，使用 ori_texts 替换 replaced_texts
                replaced_texts[i] = ori_texts[i]
        # 返回原始句子、修正句子、检测标签以及替换后的句子
        return list(ori_texts), list(cor_texts), det_labels, replaced_texts

class CscDatasetJson(Dataset):
    """
    "sen1": '[SEN1]'+src_sen.replace('##', ''),
    "sen2": '[SEN2]'+predict_text.replace('##', ''),
    "cor": tgt_sen.replace('##', ''),
    "label": label
    """

    def __init__(self, file_path):
        self.data = json.load(open(file_path, 'r', encoding='utf-8'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], \
            self.data[index]['correct_text'], self.data[index]['wrong_ids']


class CscDataset(Dataset):
    """
    "sen1": '[SEN1]'+src_sen.replace('##', ''),
    "sen2": '[SEN2]'+predict_text.replace('##', ''),
    "cor": tgt_sen.replace('##', ''),
    "label": label
    """

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, sep='\t', encoding='utf-8', header=None,
                                names=['index', 'ori', 'tgt'])
        self.data.dropna(inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        wrong_ids = get_wrong_ids(self.data.iloc[index]['ori'], self.data.iloc[index]['tgt'])
        return self.data.iloc[index]['ori'], \
            self.data.iloc[index]['tgt'], wrong_ids



def make_loaders(collate_fn, train_path='', valid_path='', test_path='',
                 batch_size=32, num_workers=4):
    train_loader = None
    if train_path and os.path.exists(train_path):
        train_loader = DataLoader(CscDataset(train_path),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    valid_loader = None
    if valid_path and os.path.exists(valid_path):
        valid_loader = DataLoader(CscDataset(valid_path),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    test_loader = None
    if test_path and os.path.exists(test_path):
        test_loader = DataLoader(CscDataset(test_path),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader


def make_json_loaders(collate_fn, train_path='', valid_path='', test_path='',
                      batch_size=32, num_workers=4):
    train_loader = None
    if train_path and os.path.exists(train_path):
        train_loader = DataLoader(CscDatasetJson(train_path),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    valid_loader = None
    if valid_path and os.path.exists(valid_path):
        valid_loader = DataLoader(CscDatasetJson(valid_path),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    test_loader = None
    if test_path and os.path.exists(test_path):
        test_loader = DataLoader(CscDatasetJson(test_path),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    model_name = 'bert-base-chinese'
    batch_size = 128

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    collator = RandomReplDataCollator(tokenizer)

    # 加载数据
    train_loader, valid_loader, test_loader = make_loaders(collator,
                                                                 train_path='/data/datasets/cscd-ns/lcsts-ime-2m.tsv',
                                                                 batch_size=batch_size, num_workers=32)

    for data_batch in tqdm.tqdm(train_loader):
        oris, cors, det_labels, replaces = data_batch
        for x, y in zip(replaces, cors):
            try:
                assert len(tokenizer(x)['input_ids']) == len(tokenizer(y)['input_ids'])
            except:
                print(x, y)
