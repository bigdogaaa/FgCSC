import pandas as pd
import json
import evaluate
import torch

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def get_perplexity(sens):
    perplexity = evaluate.load("perplexity", module_type="metric")

    results = perplexity.compute(model_id='gpt2',
                                 add_start_token=False,
                                 predictions=sens,
                                 device=device)
    return results["perplexities"]


if __name__ == '__main__':
    while 1:
        text = input("输入样本：")
        print(get_perplexity([text]))



