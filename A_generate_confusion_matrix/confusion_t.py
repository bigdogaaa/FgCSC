import json

import numpy as np
from transformers import BertTokenizer
from scipy.sparse import csr_matrix
import os
import pickle as pkl


import numpy as np
from tqdm import tqdm

"""
Section 3.2 Masking strategy
"""
def construct_confusion_matrix(confusion_data, tokenizer, temperature=1.0):
    vocab_size = tokenizer.vocab_size
    confusion_matrix = np.zeros((vocab_size, vocab_size))

    # Step 1: Populate the confusion matrix
    for key, values in tqdm(confusion_data.items()):
        key_id = tokenizer.convert_tokens_to_ids(key)
        if key_id == tokenizer.unk_token_id:
            continue
        for value in values:
            value_id = tokenizer.convert_tokens_to_ids(value)
            if value_id == tokenizer.unk_token_id:
                continue
            confusion_matrix[key_id, value_id] += 1

    # Step 2~4: Per-row processing (normalize -> flip -> normalize again)
    confusion_matrix_final = np.zeros_like(confusion_matrix)

    for i in range(vocab_size):
        row = confusion_matrix[i]
        nonzero_indices = np.where(row > 0)[0]
        if len(nonzero_indices) == 0:
            continue  # skip all-zero rows

        # Step 2: Normalize over nonzero values (linearly)
        row_values = row[nonzero_indices]
        row_sum = row_values.sum()
        if row_sum < 1e-12:
            continue
        row_normalized = row_values / row_sum


        # Step 3: Flip the probabilities
        row_flipped = 1.0 - row_normalized

        # Step 4: Normalize flipped values
        flipped_sum = row_flipped.sum()
        row_flipped = row_flipped / flipped_sum if flipped_sum > 1e-12 else np.zeros_like(row_flipped)


        # Assign back to matrix
        confusion_matrix_final[i, nonzero_indices] = row_flipped

    # Step 5: Validate
    row_sums = confusion_matrix_final.sum(axis=1)
    tolerance = 1e-8
    not_normalized_rows = np.where((row_sums > 0) & (np.abs(row_sums - 1.0) > tolerance))[0]

    if len(not_normalized_rows) == 0:
        print("✅ 所有非零行的翻转归一化后行和约等于 1。")
    else:
        print(f"⚠️ 有 {len(not_normalized_rows)} 行翻转后行和不为 1。例如前几个行索引:", not_normalized_rows[:10])
        print("对应的行和:", row_sums[not_normalized_rows[:10]])

    return confusion_matrix_final



current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'sighan'
matrix_path = f'{current_dir}/../static/confus_all_add_{dataset_name}.pkl'
if not os.path.exists(matrix_path):
    # Example usage
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
    confusion_data = json.load(open(current_dir+f'/../static/confus_all_add_{dataset_name}.json', 'r', encoding='utf-8'))

    confusion_matrix = construct_confusion_matrix(confusion_data, tokenizer, temperature=0.5)
    pkl.dump(confusion_matrix, open(matrix_path, 'wb'))
else:
    confusion_matrix = pkl.load(open(matrix_path, 'rb'))
    # print(confusion_matrix)
    sparse_matrix = csr_matrix(confusion_matrix)
    # 打印每一行中概率最高的前 N 个项
    top_n = 100  # 可根据需要修改

    # 获取 id -> token 的映射
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
    id2token = {v: k for k, v in tokenizer.vocab.items()}

    for row_idx in range(confusion_matrix.shape[0]):
        row = confusion_matrix[row_idx]
        if np.sum(row) == 0:
            continue  # 跳过全零行

        top_indices = np.argsort(row)[-top_n:][::-1]
        top_probs = row[top_indices]

        tokens = [id2token.get(idx, '[UNK]') for idx in top_indices]
        if id2token.get(row_idx, '[UNK]') == '的':
            print(f"【{id2token.get(row_idx, '[UNK]')}】 →", end=" ")
            for token, prob in zip(tokens, top_probs):
                print(f"{token}:{prob:.4f}", end=" ")
            print()


