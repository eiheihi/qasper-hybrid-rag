"""Create document-level routing profiles for the validation corpus."""

import json
from pathlib import Path

from datasets import load_dataset


PAPER_LIMIT = 100
OUTPUT_PATH = Path("data/qasper/paper_profiles.jsonl")


def normalize(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return " ".join(str(value or "").split())


def main():
    dataset = load_dataset("allenai/qasper", revision="refs/convert/parquet")
    papers = dataset["validation"].select(range(PAPER_LIMIT))
    profiles = []

    for paper in papers:
        full_text = paper["full_text"]
        section_names = full_text["section_name"]
        paragraphs_by_section = full_text["paragraphs"]
        introduction_parts = []
        for section_name, paragraphs in zip(section_names, paragraphs_by_section):
            if "introduction" in normalize(section_name).lower():
                introduction_parts.extend(normalize(paragraph) for paragraph in paragraphs[:2])
        introduction = " ".join(part for part in introduction_parts if part)
        section_list = ", ".join(normalize(name) for name in section_names if normalize(name))
        title = normalize(paper["title"])
        abstract = normalize(paper.get("abstract", ""))
        profile_text = (
            f"Paper title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Sections: {section_list}\n"
            f"Introduction excerpt: {introduction}"
        )
        profiles.append(
            {
                "paper_id": paper["id"],
                "title": title,
                "abstract": abstract,
                "profile_text": profile_text,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for profile in profiles:
            file.write(json.dumps(profile, ensure_ascii=False) + "\n")
    print(f"论文级路由 profile 已生成：{len(profiles)} 篇")
    print(f"保存位置：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
