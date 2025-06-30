# -*- coding: utf-8 -*-

import argparse
import os
import sys

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizerFast


"""
python sota.py --config_file macbert4csc.yml
"""


sys.path.append('../..')

from data.basis import make_loaders, DataCollator
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from models.mymodel import MacBert4Csc
from utils.modified_cfg import cfg

cfg.MODEL.TOKENIZER_PATH = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"


def args_parse(config_file=''):
    parser = argparse.ArgumentParser(description="csc")
    parser.add_argument(
        "--config_file", default="S2.yml", help="path to config file", type=str
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


def main():
    cfg = args_parse()
    logger.info(f'load model, model arch: {cfg.MODEL.NAME}')
    if cfg.MODEL.TOKENIZER_PATH == '':
        tokenizer = BertTokenizerFast.from_pretrained(cfg.MODEL.BERT_CKPT)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(cfg.MODEL.TOKENIZER_PATH)
    collator = DataCollator(tokenizer=tokenizer)
    # 加载数据
    train_loader, valid_loader, test_loader = make_loaders(
    # train_loader, valid_loader, test_loader = make_json_loaders(
        collator, train_path=cfg.DATASETS.TRAIN,
        valid_path=cfg.DATASETS.VALID, test_path=cfg.DATASETS.TEST,
        batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=4
    )
    # 加载之前保存的模型，继续训练
    if cfg.MODEL.WEIGHTS != '':
        if cfg.MODEL.NAME == 'macbert4csc':
            model = MacBert4Csc.load_from_checkpoint(
                checkpoint_path=cfg.MODEL.WEIGHTS,
                cfg=cfg,
                map_location=device,
                tokenizer=tokenizer
            )
        else:
            raise ValueError("model not found.")
    else:
        if cfg.MODEL.NAME == 'macbert4csc':
            model = MacBert4Csc(cfg, tokenizer)
        else:
            raise ValueError("model not found.")

    # 配置模型保存参数
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        monitor='f1',
        dirpath=cfg.OUTPUT_DIR,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='max'
    )
    # 配置模型保存参数和日志记录
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    csv_logger = CSVLogger(save_dir=os.path.join(cfg.OUTPUT_DIR, "logs"), name="training_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=os.path.join(cfg.OUTPUT_DIR, "logs"), name="training_logs")

    # 训练模型
    logger.info('training model ...')
    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        devices=None if device == torch.device('cpu') else cfg.MODEL.GPU_IDS,
        accumulate_grad_batches=cfg.SOLVER.ACCUMULATE_GRAD_BATCHES,
        callbacks=[ckpt_callback],
        logger=[csv_logger, tensorboard_logger]
    )
    # 进行训练
    torch.autograd.set_detect_anomaly(True)
    if 'train' in cfg.MODE and train_loader and len(train_loader) > 0:
        if valid_loader and len(valid_loader) > 0:
            trainer.fit(model, train_loader, valid_loader)
        else:
            trainer.fit(model, train_loader)
        logger.info('train model done.')
    # 进行测试的逻辑同训练
    if 'test' in cfg.MODE and test_loader and len(test_loader) > 0:
        # trainer = pl.Trainer(
        #     devices=1,  # 强制使用单卡
        #     accelerator="gpu"
        # )
        # model = MacBert4Csc.load_from_checkpoint(checkpoint_path=ckpt_callback.best_model_path)
        trainer.test(model, test_loader)
    # 模型转为transformers可加载
    if ckpt_callback and len(ckpt_callback.best_model_path) > 0:
        ckpt_path = ckpt_callback.best_model_path
    else:
        ckpt_path = ''
    logger.info(f'ckpt_path: {ckpt_path}')
    if ckpt_path and os.path.exists(ckpt_path):
        tokenizer.save_pretrained(cfg.OUTPUT_DIR)


if __name__ == '__main__':
    main()
