#!/usr/bin/env python
# encoding: utf-8

from judge.util import input_check_and_process, compute_p_r_f1, write_report


def calculate_metric(src_sentences, tgt_sentences, pred_sentences, report_file=None, ignore_chars="", is_token=False):
    """
    :param src_sentences: list of origin sentences
    :param tgt_sentences: list of target sentences
    :param pred_sentences: list of predict sentences
    :param report_file: report file path
    :param ignore_chars: chars that is not evaluated
    :return:
    """
    src_char_list, tgt_char_list, pred_char_list = input_check_and_process(src_sentences, tgt_sentences, pred_sentences)
    sentence_detection, sentence_correction, char_detection, char_correction = [
        {'all_error': 0, 'true_predict': 0, 'all_predict': 0} for _ in range(4)]
    output_errors = ['', []]
    for src_chars, tgt_chars, pred_chars in zip(src_char_list, tgt_char_list, pred_char_list):
        true_error_indexes = []
        detect_indexes = []
        for index, (src_char, tgt_char, pred_char) in enumerate(zip(src_chars, tgt_chars, pred_chars)):
            # if src_char in ignore_chars:
            #     src_chars[index] = tgt_char
            #     pred_chars[index] = tgt_char
            #     continue
            if src_char != tgt_char:
                char_detection['all_error'] += 1
                char_correction['all_error'] += 1
                true_error_indexes.append(index)
            if src_char != pred_char:
                char_detection['all_predict'] += 1
                char_correction['all_predict'] += 1
                detect_indexes.append(index)
                if src_char != tgt_char:
                    char_detection['true_predict'] += 1
                if pred_char == tgt_char:
                    char_correction['true_predict'] += 1
        if true_error_indexes:
            sentence_detection['all_error'] += 1
            sentence_correction['all_error'] += 1
        if detect_indexes:
            sentence_detection['all_predict'] += 1
            sentence_correction['all_predict'] += 1
            if tuple(true_error_indexes) == tuple(detect_indexes):
                sentence_detection['true_predict'] += 1
            if tuple(tgt_chars) == tuple(pred_chars):
                sentence_correction['true_predict'] += 1
        if is_token:
            if tuple(tgt_chars) != tuple(pred_chars):
                origin_s = "".join(src_chars)
                target_s = "".join(tgt_chars)
                predict_s = "".join(pred_chars)
                if origin_s == target_s and origin_s != predict_s:
                    error_type = "过纠"
                elif origin_s != target_s and origin_s == predict_s:
                    error_type = "漏纠"
                else:
                    error_type = '综合'
                output_errors[1].append(
                    [
                        "原始: " + "".join(src_chars),
                        "正确: " + "".join([c2 if c1 == c2 else f"【{c2}】" for c1, c2 in zip(pred_chars, tgt_chars)]),
                        "预测: " + "".join([c1 if c1 == c2 else f"【{c1}】" for c1, c2 in zip(pred_chars, tgt_chars)]),
                        "错误类型: " + error_type,
                        # "纠错过程：" + "aaa"
                        "纠错过程：" + "".join(["" if c1 == c2 else f"{c1}->{c2} " for c1, c2 in zip(src_chars, pred_chars)])
                    ]
                )

    result = dict()
    for prefix_name, sub_metric in zip(['sentence_level_detection_', 'sentence_level_correction_',
                                        'char_level_detection_', 'char_level_correction_'],
                                       [sentence_detection, sentence_correction, char_detection, char_correction]):
        sub_r = compute_p_r_f1(sub_metric['true_predict'], sub_metric['all_predict'], sub_metric['all_error']).items()
        for k, v in sub_r:
            result[prefix_name + k] = v
    if report_file:
        output_errors[0] = 'error/total: {}/{}'.format(len(output_errors[1]), len(src_sentences))
        write_report(report_file, result, output_errors)
    return result

