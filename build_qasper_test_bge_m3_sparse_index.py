import json
from pathlib import Path

from bge_m3_sparse import BGEM3SparseEncoder, build_inverted_index
from tune_qasper_bge_m3_hybrid import read_jsonl


CORPUS_PATH = Path("data/qasper_test/corpus.jsonl")
INDEX_PATH = Path("data/qasper_test/bge_m3_sparse_index.json")
ENCODE_BATCH_SIZE = 4


def main():
    corpus = read_jsonl(CORPUS_PATH)
    if not corpus:
        raise RuntimeError("语料为空，请先运行 prepare_qasper_test.py")

    print(f"加载 BGE-M3 原生稀疏编码器，待编码 test 文本块：{len(corpus)}")
    encoder = BGEM3SparseEncoder(batch_size=ENCODE_BATCH_SIZE, max_length=512)
    lexical_weights = []

    for start in range(0, len(corpus), ENCODE_BATCH_SIZE):
        batch = corpus[start : start + ENCODE_BATCH_SIZE]
        lexical_weights.extend(encoder.encode([item["text"] for item in batch]))
        completed = min(start + ENCODE_BATCH_SIZE, len(corpus))
        if completed % 100 == 0 or completed == len(corpus):
            print(f"已编码：{completed}/{len(corpus)}")

    inverted_index = build_inverted_index(lexical_weights)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(
            {
                "model": "BAAI/bge-m3",
                "corpus_size": len(corpus),
                "max_length": 512,
                "inverted_index": inverted_index,
            }
        ),
        encoding="utf-8",
    )
    print(f"\nBGE-M3 test 原生稀疏索引已完成：{len(inverted_index)} 个词项")
    print(f"保存位置：{INDEX_PATH}")


if __name__ == "__main__":
    main()
