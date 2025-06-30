# Description 
Source code for PRICAI 2025 submission 532: "Fine-Grained Confidence Estimation to Mitigating Suboptimal Corrections in Chinese Spelling Correction".

# Candidate Generation

## Catalog
| Index | Function                   | Related Thesis Section                             | Corresponding Package    |
|-------|----------------------------|----------------------------------------------------|--------------------------|
| 1     | Model Architecture         | Section 3.2 (Two-stage Training)                   | models.mymodel           |
| 2     | Model Training & Inferring | Section 3.2 (Two-stage Training)                   | candidate_generation.*   |
| 3     | Confidence Estimation      | Section 3.3 (Candidate Comparison)                 | reranking.1-4            |
| 4     | Candidate Re-ranking       | Section 3.4 (Candidate Compression and Re-Ranking) | reranking.5-6 (updating) |
>The LLM reranking prompts are being organized.

# Quick Start

## 1. Download dataset
| Index | Dataset  | url                                     |
|-------|----------|-----------------------------------------|
| 1     | CSCD-NS  | https://github.com/nghuyong/cscd-ns     |
| 2     | SIGHAN15 | https://github.com/liushulinle/CRASpell |


## 2. Train model
1. python S1.py
2. python S2.py
3. python topk_predict_upon_limit.py
>Each step needs to modify corresponding yml file.

## 3. Re-ranking
1. Generate scores (1-6 in reranking package)
2. final decision made by LLMs (codes are under organizing)

