# QASPER Hybrid RAG

面向科学论文问答的两阶段检索增强生成（RAG）项目。项目基于 QASPER 公开数据集，比较 Dense Retrieval、BM25-Dense Hybrid Retrieval 与 Cross-Encoder Reranking，并使用带证据标注的问题进行可复现评测。

## Pipeline

```text
Question
  ├─ Dense Retrieval (Top-30)
  ├─ BM25 Retrieval (Top-30)
  └─ Weighted RRF Fusion (Top-20)
              ↓
   BAAI/bge-reranker-v2-m3
              ↓
        Top-10 Evidence Chunks
              ↓
       DeepSeek Answer Generation
```

## Evaluation Setup

- Dataset: QASPER scientific paper QA dataset
- Validation corpus: 100 papers / 4,475 text chunks
- Evaluation questions: 253 questions with gold evidence spans
- Metrics: Hit@k, Recall@k, MRR@10, nDCG@10, P95 latency
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU

## Validation Results

| System | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | P95 Latency |
|---|---:|---:|---:|---:|---:|
| Dense Retrieval | 9.49% | 20.16% | 0.1409 | 0.1271 | 11.5 ms |
| Weighted RRF | 13.83% | 24.11% | 0.1950 | 0.1732 | 22.0 ms |
| Weighted RRF + BGE Reranker | **16.21%** | **34.78%** | **0.2372** | **0.2118** | **501.8 ms** |

The final two-stage retriever improves Hit@5 by 14.62 percentage points over the dense baseline.

## Key Design

- BM25 supplies lexical matches that dense retrieval may miss.
- Weighted RRF combines BM25 and Dense rankings.
- The selected RRF setting uses BM25 weight `1.0`, Dense weight `0.25`, and `RRF_K=60`.
- `BAAI/bge-reranker-v2-m3` reranks the top 20 fused candidates on GPU.
- All comparison systems use the same QASPER validation questions and evidence labels.

## Run

```powershell
python prepare_qasper.py
python build_qasper_index.py
python evaluate_dense.py
python tune_hybrid_rrf.py
python evaluate_hybrid_rerank.py
```

## Project Structure

```text
prepare_qasper.py              # Prepare QASPER corpus and evidence QA pairs
build_qasper_index.py          # Build Chroma dense index
evaluate_dense.py              # Dense retrieval baseline
tune_hybrid_rrf.py             # BM25 + Dense weighted RRF experiments
evaluate_hybrid_rerank.py      # Two-stage Hybrid + Reranker evaluation
main.py                        # FastAPI prototype
rag_service.py                 # DeepSeek RAG service
```

## Security

`.env`, local models, vector databases, downloaded datasets, and generated caches are excluded from Git. Create a local `.env` file with your own DeepSeek API key before running generation-related features.