# -*- coding: utf-8 -*-

import argparse
import os
import sys

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizerFast
import pandas as pd


topk = 5

"""
python sota.py --config_file macbert4csc.yml
"""


sys.path.append('../..')

from data.basis import make_loaders, DataCollator
from models.mymodel import MacBert4Csc
from pycorrector.macbert.defaults import _C as cfg
from judge.evaluate import calculate_metric
from utils.verify_data import predict_print_fmt
from utils.model_output_process import convert_token_ids2sentence
from tqdm import tqdm


cfg.MODEL.TOKENIZER_PATH = ''

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"

from utils.model_random_sed import set_seed

# 设置随机种子
set_seed(42)


def args_parse(config_file=''):
    parser = argparse.ArgumentParser(description="csc")
    parser.add_argument(
        "--config_file", default="macbert4csc_predict.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--opts", help="Modify config options using the command-line key value", default=[],
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    logger.info(args)
    config_file = args.config_file or config_file
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if config_file != '':
        logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    return cfg


if __name__ == '__main__':
    cfg = args_parse()
    device = torch.device(f"cuda:{cfg.MODEL.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    if cfg.MODEL.TOKENIZER_PATH == '':
        tokenizer = BertTokenizerFast.from_pretrained(cfg.MODEL.BERT_CKPT)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(cfg.MODEL.TOKENIZER_PATH)
    if cfg.MODEL.WEIGHTS != '':
        logger.info(f'loading model {cfg.MODEL.WEIGHTS}')
        model = MacBert4Csc.load_from_checkpoint(
            checkpoint_path=cfg.MODEL.WEIGHTS,
            cfg=cfg,
            map_location=device,
            tokenizer=tokenizer
        )
        model.eval()
        collator = DataCollator(tokenizer=tokenizer)
        train_loader, valid_loader, test_loader = make_loaders(
            collator, train_path='',
            valid_path='', test_path=cfg.DATASETS.TEST,
            batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=4
        )
        results = []
        print_results = []
        out_l = []
        for batch in tqdm(test_loader, total=len(test_loader)):
            texts = batch[0]
            gts = batch[1]
            det_labels = batch[2]
            # rst = model.det_predict(texts)
            rst, det_rst, topk_rst = model.new_predict(texts, topk=topk)

            for src_text, tgt_text, det_label, pre_token_idss in zip(texts, gts, det_labels, topk_rst):
                # filtered_det_label = det_label[1:1 + len(pre)].cpu().tolist()
                _src = tokenizer(src_text, add_special_tokens=False)['input_ids']
                _tgt = tokenizer(tgt_text, add_special_tokens=False)['input_ids']
                find = False
                for pre_token_ids in pre_token_idss:
                    if pre_token_ids == _tgt:
                        find = True
                        _pre = pre_token_ids
                        break
                if not find:
                    _pre = pre_token_idss[0]
                # if _src not in pre_token_idss:
                #     pre_token_idss.append(_src)
                results.append((_src, _tgt, _pre))
                for id_, (x,y,z) in enumerate(zip(_src, _tgt, _pre)):
                    if x == tokenizer.unk_token_id:
                        _src[id_] = y
                _src_text = convert_token_ids2sentence(_src, tokenizer)
                _tgt_text = convert_token_ids2sentence(_tgt, tokenizer)
                _pre_text = convert_token_ids2sentence(_pre, tokenizer)
                print_results.append((_src_text, _tgt_text, _pre_text))

                _pre_texts = []
                for _ in pre_token_idss:
                    _pre_texts.append(convert_token_ids2sentence(_, tokenizer))
                if len(_src_text) == len(_tgt_text):
                    out_l.append((_src_text, _tgt_text, _pre_texts))
        src_sentences, tgt_sentences, pre_sentences = zip(*print_results)
        predict_print_fmt(src_sentences, tgt_sentences, pre_sentences, './predict_result.txt')
        src_sentences, tgt_sentences, pre_sentences = zip(*results)
        print(calculate_metric(src_sentences, tgt_sentences, pre_sentences))

        out_df = pd.DataFrame.from_records(out_l)
        print(out_df)
        out_df.to_csv(f'{cfg.DATASETS.TEST}.pre_sota_topk',encoding='utf-8', header=False, index=False)
    else:
        raise Exception('未加载模型权重！')


