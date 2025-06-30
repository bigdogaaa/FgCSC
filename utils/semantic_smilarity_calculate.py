from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

model_name='/data/llm/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name)
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def get_semantic_similarity(sen1, sen2s):
    """
    计算两个句子的语义相似度

    参数:
        sen1 (str): 第一个句子
        sen2 (str): 第二个句子
        model_name (str): 用于生成句子嵌入的预训练模型名称，默认使用'all-MiniLM-L6-v2'

    返回:
        float: 两个句子的语义相似度分数，范围在0到1之间
    """
    # 加载预训练模型


    # 生成句子的嵌入向量
    embeddings1 = model.encode([sen1])
    embeddings2 = model.encode(sen2s)

    # 计算余弦相似度
    similarities = cosine_similarity([embeddings1[0]], embeddings2)[0]

    return similarities


if __name__ == '__main__':
    while True:
        sen1 = input("输入第一个句子：")
        sen2 = input("输入第二个句子：")
        similarities = get_semantic_similarity(sen1, [sen2])
        print(f"语义相似度：{similarities}")