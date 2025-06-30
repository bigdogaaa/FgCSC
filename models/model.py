# -*- coding: utf-8 -*-


from abc import ABC
from loguru import logger
import numpy as np
import pytorch_lightning as pl
from pycorrector.macbert import lr_scheduler

from judge.evaluate import calculate_metric


import torch
import torch.nn as nn

from utils.model_random_sed import set_seed


# 设置随机种子
set_seed(42)


class FlattenFocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss.
    copy from https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FlattenFocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, sequence_length, num_cls] or [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size, sequence_length] or [batch_size]
        Returns:
            shape of [batch_size, sequence_length] or [batch_size]
        """
        if self.activation_type == 'softmax':
            # Reshape input and target if they have sequence_length dimension
            if input.dim() == 3:  # [batch_size, sequence_length, num_cls]
                batch_size, sequence_length, num_cls = input.size()
                input = input.view(-1, num_cls)  # Flatten to [batch_size * sequence_length, num_cls]
                target = target.view(-1)  # Flatten to [batch_size * sequence_length]
            else:  # [batch_size, num_cls]
                batch_size, num_cls = input.size()
                sequence_length = 1  # No sequence dimension

            # Convert target to one-hot encoding
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)

            # Compute softmax and focal loss
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)  # Sum over classes

            # Reshape loss back to [batch_size, sequence_length] if necessary
            if sequence_length > 1:
                loss = loss.view(batch_size, sequence_length)

        elif self.activation_type == 'sigmoid':
            # Sigmoid case remains unchanged
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        # return loss
        return loss.sum(dim=-1)  # Return sum loss over sequence per batch

class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss.
    copy from https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    scheduler_args = {
        "optimizer": optimizer,

        # warmup options
        "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
        "warmup_epochs": cfg.SOLVER.WARMUP_EPOCHS,
        "warmup_method": cfg.SOLVER.WARMUP_METHOD,

        # multi-step lr scheduler options
        "milestones": cfg.SOLVER.STEPS,
        "gamma": cfg.SOLVER.GAMMA,

        # cosine annealing lr scheduler options
        "max_iters": cfg.SOLVER.MAX_ITER,
        "delay_iters": cfg.SOLVER.DELAY_ITERS,
        "eta_min_lr": cfg.SOLVER.ETA_MIN_LR,

    }
    scheduler = getattr(lr_scheduler, cfg.SOLVER.SCHED)(**scheduler_args)
    return {'scheduler': scheduler, 'interval': cfg.SOLVER.INTERVAL}


class BaseTrainingEngine(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def configure_optimizers(self):
        optimizer = make_optimizer(self.cfg, self)
        scheduler = build_lr_scheduler(self.cfg, optimizer)

        return [optimizer], [scheduler]

    def on_validation_epoch_start(self) -> None:
        logger.info('Valid.')

    def on_test_epoch_start(self) -> None:
        logger.info('Testing...')


class CscTrainingModel(BaseTrainingEngine, ABC):
    """
        用于CSC的BaseModel, 定义了训练及预测步骤
        """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # loss weight
        self.w = cfg.MODEL.HYPER_PARAMS[0]
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(ori_text), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ori_texts, cor_texts, det_labels = batch
        outputs = self.forward(ori_texts, cor_texts, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_texts, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []

        for src, tgt, predict, det_predict, det_label in zip(ori_texts, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()

            results.append((_src, _tgt, _predict))

        self.validation_step_outputs.append((loss.cpu().item(), results))
        return loss.cpu().item(), results

    def on_validation_epoch_end(self):
        results = []
        for out in self.validation_step_outputs:
            results += out[1]
        loss = np.mean([out[0] for out in self.validation_step_outputs])
        self.log('val_loss', loss, sync_dist=True)
        logger.info(f'loss: {loss}')
        src_sentences, tgt_sentences, pre_sentences = zip(*results)
        metric = calculate_metric(src_sentences, tgt_sentences, pre_sentences)
        self.log('f1', metric['char_level_correction_f1'], sync_dist=True)
        logger.info(f'Char-detection:  precision: {metric['char_level_detection_precision']}')
        logger.info(f'Char-detection:  recall: {metric['char_level_detection_recall']}')
        logger.info(f'Char-detection:  f1: {metric['char_level_detection_f1']}')
        logger.info(f'Char-correction:  precision: {metric['char_level_correction_precision']}')
        logger.info(f'Char-correction:  recall: {metric['char_level_correction_recall']}')
        logger.info(f'Char-correction:  f1: {metric['char_level_correction_f1']}')
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        ori_texts, cor_texts, det_labels = batch
        outputs = self.forward(ori_texts, cor_texts, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_texts, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []

        for src, tgt, predict, det_predict, det_label in zip(ori_texts, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()


            results.append((_src, _tgt, _predict))
        self.test_step_outputs.append((loss.cpu().item(), results))
        return loss.cpu().item(), results

    def on_test_epoch_end(self) -> None:
        logger.info('Test.')

        results = []
        for out in self.test_step_outputs:
            results += out[1]
        loss = np.mean([out[0] for out in self.test_step_outputs])
        src_sentences, tgt_sentences, pre_sentences = zip(*results)
        metric = calculate_metric(src_sentences, tgt_sentences, pre_sentences)
        logger.info(f'Char-detection:  precision: {metric['char_level_detection_precision']}')
        logger.info(f'Char-detection:  recall: {metric['char_level_detection_recall']}')
        logger.info(f'Char-detection:  f1: {metric['char_level_detection_f1']}')
        logger.info(f'Char-correction:  precision: {metric['char_level_correction_precision']}')
        logger.info(f'Char-correction:  recall: {metric['char_level_correction_recall']}')
        logger.info(f'Char-correction:  f1: {metric['char_level_correction_f1']}')
        self.test_step_outputs.clear()

    def det_predict(self, texts):
        rst = []
        with torch.no_grad():
            # 检错输出，纠错输出
            det_output, cor_output = self.forward(texts)
            det_y_hat = (det_output > 0.5).long()
            for det_predict, src in zip(det_y_hat, texts):
                tmp = []
                _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
                pre = det_predict[1:len(_src) + 1].cpu().numpy().tolist()
                for mask_val, src_val in zip(pre, _src):
                    if mask_val == 1:
                        # tmp.append(self.tokenizer.unk_token_id)
                        tmp.append(-10000)
                    else:
                        tmp.append(src_val)
                rst.append(tmp)
                # rst.append(self.tokenizer.decode(tmp).replace(' ', '').replace('[UNK]', '‽').replace('##', ''))
        return rst

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        inputs.to(self.cfg.MODEL.DEVICE)
        with torch.no_grad():
            outputs = self.forward(texts)
            det_y_hat = (outputs[0] > 0.5).long()
            y_hat = torch.argmax(outputs[1], dim=-1)
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
        rst = []
        det_rst = []
        for t_len, _y_hat, det_predict, src in zip(expand_text_lens, y_hat, det_y_hat, texts):
            rst.append(_y_hat[1:t_len])
            tmp = []
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            pre = det_predict[1:t_len].cpu().numpy().tolist()
            for mask_val, src_val in zip(pre, _src):
                if mask_val == 1:
                    # tmp.append(self.tokenizer.unk_token_id)
                    tmp.append(-10000)
                else:
                    tmp.append(src_val)
                det_rst.append(tmp)
            # rst.append(self.tokenizer.decode(_y_hat[1:t_len]).replace(' ', '').replace('[UNK]', '‽').replace('##', ''))
        return rst, det_rst

    def new_predict(self, texts, topk=5):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt', truncation=True)
        inputs.to(self.cfg.MODEL.DEVICE)
        with torch.no_grad():
            outputs = self.forward(texts)
            det_outputs = outputs[0]
            logits = outputs[1]

            det_y_hat = (det_outputs > 0.5).long()
            y_hat = torch.argmax(logits, dim=-1)

            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1

        batch_size = len(texts)
        rst = []
        det_rst = []
        candidate_rst = []

        for i in range(batch_size):
            t_len = expand_text_lens[i].item()
            _y_hat = y_hat[i][1:t_len]
            _det_predict = det_y_hat[i][1:t_len]
            src_tokens = self.tokenizer(texts[i], add_special_tokens=False)['input_ids']
            src_tokens = src_tokens[:t_len - 1]  # Remove CLS token and possibly others

            # Get the logits for current sample
            current_logits = logits[i][1:t_len, :]
            current_probs = torch.softmax(current_logits, dim=-1)  # Convert logits to probabilities

            # Find positions where _y_hat and src_tokens differ
            diff_indices = [j for j, (h, s) in enumerate(zip(_y_hat, src_tokens)) if h != s]

            # Prepare beam search for these positions
            beam_width = max(topk * 2, 3)  # Adjust beam width as needed
            beam = [([], 1.0)]  # Each element is a tuple of (sequence, -log_prob)

            for idx in diff_indices:
                new_beam = []
                for path, log_prob in beam:
                    # Get probabilities for current position
                    prob_dist = current_probs[idx].cpu().numpy()
                    # Get topk candidates
                    top_candidates = np.argsort(-prob_dist)[:beam_width]
                    for tok in top_candidates:
                        new_path = path + [tok]
                        new_log_prob = log_prob - np.log(prob_dist[tok] + 1e-20)
                        new_beam.append((new_path, new_log_prob))
                # Keep top beam_width paths
                new_beam.sort(key=lambda x: x[1])
                beam = new_beam[:beam_width]

            # Generate topk sequences
            topk_sequences = []
            for k in range(min(topk, len(beam))):
                seq, _ = beam[k]
                # Create new y_hat with beam search tokens inserted
                new_y_hat = _y_hat.tolist().copy()
                for j, tok in zip(diff_indices, seq):
                    new_y_hat[j] = tok
                topk_sequences.append(new_y_hat)

            # Decode all topk sequences
            # decoded_candidates = []
            # for seq in topk_sequences:
            #     decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
            #     decoded_candidates.append(decoded)


            # Clean up decoded candidates (remove duplicates or fix specific cases)
            # For example, removing '[UNK]' or replacing it with a placeholder

            # Append results
            rst.append(_y_hat.tolist())
            det_rst.append(_det_predict.cpu().numpy().tolist())
            candidate_rst.append(topk_sequences)

        return rst, det_rst, candidate_rst


class CscRandomTrainingModel(CscTrainingModel, ABC):
    """
        用于CSC的BaseModel, 定义了训练及预测步骤
        """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # loss weight
        self.w = cfg.MODEL.HYPER_PARAMS[0]
        self.register_buffer('ce_loss_ema', None)  # 持久化EMA值
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # 添加分阶段训练相关参数
        self.stage_start_epoch = cfg.MODEL.STAGE_START_EPOCH  # 开始引入 det_loss 的阶段
        self.stage_max_epoch = cfg.MODEL.STAGE_MAX_EPOCH  # det_loss 达到最大权重的阶段
        self.det_loss_max_weight = cfg.MODEL.DET_LOSS_MAX_WEIGHT

    def training_step(self, batch, batch_idx):
        ori_texts, cor_texts, det_labels, replaced_texts = batch
        outputs = self.forward(ori_texts, cor_texts, det_labels, replaced_texts)

        # 分阶段引入 sim_loss 权重
        current_epoch = self.current_epoch
        if current_epoch < self.stage_start_epoch:
            weight_det_loss = 0.0
        elif current_epoch < self.stage_max_epoch:
            weight_det_loss = ((current_epoch - self.stage_start_epoch) /
                               (self.stage_max_epoch - self.stage_start_epoch)) * self.det_loss_max_weight
        else:
            weight_det_loss = self.det_loss_max_weight

        # 动态计算 total_cor_loss
        total_loss = self.w * outputs[1] + weight_det_loss * (1 - self.w) * outputs[0]

        # 记录损失
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=len(ori_texts), sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        ori_texts, cor_texts, det_labels, replaced_texts = batch
        outputs = self.forward(ori_texts, cor_texts, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_texts, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []

        for src, tgt, predict, det_predict, det_label in zip(ori_texts, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()

            results.append((_src, _tgt, _predict))

        self.validation_step_outputs.append((loss.cpu().item(), results))
        return loss.cpu().item(), results

    def test_step(self, batch, batch_idx):
        ori_texts, cor_texts, det_labels, replaced_texts = batch
        outputs = self.forward(ori_texts, cor_texts, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_texts, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []

        for src, tgt, predict, det_predict, det_label in zip(ori_texts, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()

            results.append((_src, _tgt, _predict))
        self.test_step_outputs.append((loss.cpu().item(), results))
        return loss.cpu().item(), results






