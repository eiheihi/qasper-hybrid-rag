# QASPER Hybrid RAG

A reproducible, evaluation-driven RAG project for scientific-paper question answering. The final global-search pipeline combines BGE-M3 dense retrieval, BGE-M3 native sparse retrieval, BM25, weighted Reciprocal Rank Fusion (RRF), and a BGE cross-encoder reranker.

## Final Global Retrieval Pipeline

```text
Question
  ├─ BGE-M3 dense retrieval (Top-30)
  ├─ BGE-M3 native sparse retrieval (Top-30)
  ├─ BM25 lexical retrieval (Top-30)
  └─ Weighted RRF fusion (Dense=2.0, Sparse=1.0, BM25=1.0)
                              ↓
                    Top-20 candidates
                              ↓
                 BAAI/bge-reranker-v2-m3
                              ↓
                 Top-5 cited evidence chunks
                              ↓
                 DeepSeek answer generation
```

`Hit@k` in this repository measures evidence retrieval: whether a gold QASPER evidence paragraph appears in the retrieved list. It is not an end-to-end answer-accuracy metric.

## Evaluation Protocol

- Dataset: [QASPER](https://huggingface.co/datasets/allenai/qasper)
- Validation: 100 papers, 4,475 paragraphs, 253 evidence-annotated questions
- Independent test: 100 papers, 4,399 paragraphs, 321 evidence-annotated questions
- Global-search setting: every question searches across all papers in the split; no ground-truth paper id is used at query time
- Metrics: Hit@k, Recall@k, MRR@10, nDCG@10, candidate recall, and P95 latency
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU

Fusion weights and reranking depth were selected on validation only. The final test run uses the frozen configuration.

## Final Results

### Validation set (model selection only)

| System | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Candidate Recall@20 |
|---|---:|---:|---:|---:|---:|
| BGE-M3 Dense | 12.65% | 34.39% | 0.2145 | 0.1962 | — |
| Dense + BM25 + RRF + reranker | 17.39% | 42.29% | 0.2680 | 0.2416 | 36.15% |
| **Dense + Sparse + BM25 + RRF + reranker** | 17.00% | **43.08%** | **0.2689** | **0.2484** | **39.12%** |

### Independent test set (frozen final configuration)

| System | Hit@1 | Hit@3 | Hit@5 | MRR@10 | nDCG@10 | Candidate Recall@20 |
|---|---:|---:|---:|---:|---:|---:|
| Dense + BM25 + RRF + reranker | 22.43% | 33.02% | 38.01% | 0.2914 | 0.2364 | 31.56% |
| **Dense + Sparse + BM25 + RRF + reranker** | **23.36%** | **34.58%** | **40.19%** | **0.3051** | **0.2480** | **33.46%** |

On the independent test set, adding BGE-M3 native sparse retrieval increases Hit@5 by **2.18 percentage points**, MRR@10 by **4.7%**, and nDCG@10 by **4.9%** over the two-way reranking baseline.

## Experimental Findings

- BGE-M3 dense embeddings substantially outperform the original MiniLM dense baseline.
- Native BGE-M3 sparse retrieval adds complementary candidates; after RRF and reranking, the improvement transfers from validation to the independent test split.
- Context-enriched passage embeddings increase candidate recall, but context-aware reranking lowers Hit@5. This branch is retained as a negative ablation and is not used by the final system.
- A paper router built from title, abstract, and introduction snippets reaches only 50.99% Recall@5, so it is not used as a hard filter in global search.

## API Demo

The FastAPI app serves the frozen global pipeline and asks DeepSeek to answer with evidence citations.

```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs`, then call:

- `GET /system` to inspect the active retrieval configuration
- `POST /ask` with `{"question": "What is the seed lexicon?"}`

The first request loads the BGE models into GPU memory, so it is slower than later requests.

## Reproduce

```powershell
# Validation: prepare corpus, build independent indexes, and tune only here
python prepare_qasper.py
python build_qasper_bge_m3_index.py
python build_qasper_bge_m3_sparse_index.py
python tune_qasper_bge_m3_three_way.py
python evaluate_qasper_three_way_rerank.py

# Independent test: use the frozen Dense=2.0 / Sparse=1.0 configuration
python prepare_qasper_test.py
python build_qasper_test_bge_m3_index.py
python build_qasper_test_bge_m3_sparse_index.py
python evaluate_qasper_test_three_way_rerank.py
```

## Project Structure

```text
bge_m3_embeddings.py                    # BGE-M3 dense adapter for Chroma
bge_m3_sparse.py                        # Native sparse encoding and inverted-index search
build_qasper_bge_m3_index.py            # Dense index builder
build_qasper_bge_m3_sparse_index.py     # Native sparse index builder
tune_qasper_bge_m3_three_way.py         # Validation-only three-way RRF sweep
evaluate_qasper_three_way_rerank.py     # Validation reranker ablation
evaluate_qasper_test_three_way_rerank.py# Frozen independent-test evaluation
qasper_rag_service.py                   # Final retrieval + DeepSeek answer service
main.py                                 # FastAPI entry point
rag_service.py                          # Earlier PDF/MiniLM prototype retained for reference
```

## Security and Large Files

`.env`, local models, vector databases, downloaded datasets, generated caches, and virtual environments are excluded from Git. Create a local `.env` containing `DEEPSEEK_API_KEY` before starting the API.
