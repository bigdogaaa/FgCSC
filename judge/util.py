#!/usr/bin/env python
# encoding: utf-8

import os.path


def compute_p_r_f1(true_predict, all_predict, all_error):
    """
    @param true_predict:
    @param all_predict:
    @param all_error:
    @return:
    """
    print('TP: %d, TP+FP: %d, TP+FN: %d' % (true_predict, all_predict, all_error))
    p = round(true_predict / (all_predict + 1e-10) * 100, 3)
    r = round(true_predict / (all_error + 1e-10) * 100, 3)
    f1 = round(2 * p * r / (p + r + 1e-10), 3)
    return {'precision': p, 'recall': r, 'f1': f1}


def write_report(output_file, metric, output_errors):
    """
    generate report
    @param output_file:
    @param metric:
    @param output_errors:
    @return:
    """
    tgt_dir = output_file.rsplit('/', 1)[0]
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    with open(output_file, 'wt', encoding='utf-8') as f:
        f.write('overview:\n')
        for key in metric:
            f.write(f'{key}:{metric[key]}\n')
        f.write('\nbad cases: {}\n'.format(output_errors[0]))
        for output_error in output_errors[1]:
            f.write("\n".join(output_error))
            f.write("\n\n")


def input_check_and_process(src_sentences, tgt_sentences, pred_sentences):
    """
    check the input is valid
    @param src_sentences:
    @param tgt_sentences:
    @param pred_sentences:
    @return:
    """
    assert len(src_sentences) == len(tgt_sentences) == len(pred_sentences)
    src_char_list = [list(s) for s in src_sentences]
    tgt_char_list = [list(s) for s in tgt_sentences]
    pred_char_list = [list(s) for s in pred_sentences]
    assert all(
        [len(src) == len(tgt) == len(pred) for src, tgt, pred in zip(src_char_list, tgt_char_list, pred_char_list)]
    )
    return src_char_list, tgt_char_list, pred_char_list
