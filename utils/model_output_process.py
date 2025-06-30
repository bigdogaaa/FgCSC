from transformers import PreTrainedTokenizer


def convert_token_ids2tokens(token_ids, tokenizer):
    # 解码每个Token ID为对应的Token字符串，禁用特殊标记和空格清理以保持准确性
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tokens


def convert_token_ids2sentence(token_ids, tokenizer):
    # 解码每个Token，禁用特殊标记和空格清理以保持准确性
    decode_args = {
        "add_special_tokens": True,
        "clean_up_tokenization_spaces": False
    }
    return tokenizer.decode(token_ids, **decode_args).replace(' ', '')

def align_token_lengths(_src: list[int], _tgt: list[int], tokenizer: PreTrainedTokenizer) -> str:
    """
    对齐源和目标Token序列的解码长度，确保每个位置解码后的字符串长度一致。

    Args:
        _src (list[int]): 源文本的Token ID列表。
        _tgt (list[int]): 目标文本的Token ID列表。
        tokenizer (PreTrainedTokenizer): 用于解码的分词器。

    Returns:
        str: 处理后的目标文本字符串，确保每个位置的解码长度与源文本一致。

    Raises:
        ValueError: 如果源和目标的Token数量不匹配。
    """
    if len(_src) != len(_tgt):
        raise ValueError("_src and _tgt must have the same length")

    # 解码每个Token，禁用特殊标记和空格清理以保持准确性
    decode_args = {
        "add_special_tokens": False,
        "clean_up_tokenization_spaces": False
    }
    src_tokens = [tokenizer.decode([tid], **decode_args) for tid in _src]
    tgt_tokens = [tokenizer.decode([tid], **decode_args) for tid in _tgt]

    # 构建新的目标Token序列
    new_tgt_ids = []
    for s_tok, t_tok, s_id, t_id in zip(src_tokens, tgt_tokens, _src, _tgt):
        if len(s_tok) != len(t_tok):
            new_tgt_ids.append(s_id)
        else:
            new_tgt_ids.append(t_id)

    # 解码为最终字符串
    aligned_tgt_text = tokenizer.decode(new_tgt_ids, **decode_args)
    return aligned_tgt_text.replace(' ', '')

