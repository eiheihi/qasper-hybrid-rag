import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

PAPER_LIMIT = 100
OUTPUT_DIR = Path("data/qasper")
CORPUS_PATH = OUTPUT_DIR / "corpus.jsonl"
EVAL_PATH = OUTPUT_DIR / "eval.jsonl"


def normalize(text: str) -> str:
    return " ".join(str(text).split())


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_answer_candidates(raw_answers):
    if isinstance(raw_answers, dict):
        answer_keys = {
            "unanswerable",
            "extractive_spans",
            "free_form_answer",
            "yes_no",
            "evidence",
        }

        if answer_keys.intersection(raw_answers):
            yield raw_answers
            return

        yield from iter_answer_candidates(raw_answers.get("answer", []))

    elif isinstance(raw_answers, list):
        for item in raw_answers:
            yield from iter_answer_candidates(item)


def answer_to_text(answer: dict) -> str:
    free_form_answer = answer.get("free_form_answer")
    if isinstance(free_form_answer, str) and free_form_answer.strip():
        return free_form_answer.strip()

    extractive_spans = answer.get("extractive_spans") or []
    if extractive_spans:
        return " ".join(extractive_spans).strip()

    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "Yes" if yes_no else "No"

    return ""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "allenai/qasper",
        revision="refs/convert/parquet",
    )

    papers = dataset["validation"].select(range(PAPER_LIMIT))

    corpus_records = []
    question_records = []
    evidence_to_chunk_ids = defaultdict(list)

    for paper in papers:
        paper_id = paper["id"]
        title = paper["title"]
        full_text = paper["full_text"]

        section_names = full_text["section_name"]
        paragraphs_by_section = full_text["paragraphs"]

        for section_index, paragraphs in enumerate(paragraphs_by_section):
            section_name = section_names[section_index] or "Untitled section"

            for paragraph_index, paragraph in enumerate(paragraphs):
                paragraph = (paragraph or "").strip()
                if not paragraph:
                    continue

                chunk_id = (
                    f"{paper_id}::section-{section_index}::paragraph-{paragraph_index}"
                )

                corpus_records.append(
                    {
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "title": title,
                        "section": section_name,
                        "text": paragraph,
                    }
                )

                evidence_to_chunk_ids[normalize(paragraph)].append(chunk_id)

        qas = paper["qas"]
        for question_index, question in enumerate(qas["question"]):
            candidates = list(
                iter_answer_candidates(qas["answers"][question_index])
            )

            gold_answers = []
            gold_evidence = []

            for candidate in candidates:
                if candidate.get("unanswerable", False):
                    continue

                answer_text = answer_to_text(candidate)
                if answer_text:
                    gold_answers.append(answer_text)

                evidence = candidate.get("evidence") or []
                if isinstance(evidence, str):
                    evidence = [evidence]

                gold_evidence.extend(
                    item for item in evidence if isinstance(item, str) and item.strip()
                )

            gold_answers = list(dict.fromkeys(gold_answers))
            gold_evidence = list(dict.fromkeys(gold_evidence))

            gold_chunk_ids = []
            for evidence_text in gold_evidence:
                gold_chunk_ids.extend(
                    evidence_to_chunk_ids.get(normalize(evidence_text), [])
                )

            gold_chunk_ids = list(dict.fromkeys(gold_chunk_ids))

            if not gold_answers or not gold_chunk_ids:
                continue

            question_records.append(
                {
                    "question_id": qas["question_id"][question_index],
                    "paper_id": paper_id,
                    "paper_title": title,
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_evidence": gold_evidence,
                    "gold_chunk_ids": gold_chunk_ids,
                }
            )

    write_jsonl(CORPUS_PATH, corpus_records)
    write_jsonl(EVAL_PATH, question_records)

    print(f"已处理论文：{len(papers)} 篇")
    print(f"语料文本块：{len(corpus_records)} 个")
    print(f"可评测问题：{len(question_records)} 条")
    print(f"语料文件：{CORPUS_PATH}")
    print(f"评测文件：{EVAL_PATH}")


if __name__ == "__main__":
    main()