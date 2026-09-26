"""Build context-enriched representations without changing retrieval labels."""

import re
from collections import defaultdict


CHUNK_ID_PATTERN = re.compile(r"^.+::section-(?P<section>\d+)::paragraph-(?P<paragraph>\d+)$")


def paragraph_position(record: dict) -> int:
    match = CHUNK_ID_PATTERN.match(record["chunk_id"])
    if not match:
        raise ValueError(f"无法从 chunk_id 解析段落位置：{record['chunk_id']}")
    return int(match.group("paragraph"))


def contextual_text_by_chunk_id(corpus: list[dict], window: int = 1) -> dict[str, str]:
    """Represent each original paragraph with stable document context.

    The returned text is only used for dense-vector encoding. Gold labels and
    the text shown to the reranker remain the original QASPER paragraph.
    """
    groups = defaultdict(list)
    for record in corpus:
        groups[(record["paper_id"], record["section"])].append(record)

    representations = {}
    for records in groups.values():
        records.sort(key=paragraph_position)
        for index, record in enumerate(records):
            before = records[max(0, index - window) : index]
            after = records[index + 1 : index + 1 + window]
            adjacent = [item["text"] for item in before + after]
            context = " ".join(adjacent)
            representations[record["chunk_id"]] = (
                f"Paper title: {record['title']}\n"
                f"Section: {record['section']}\n"
                f"Target passage: {record['text']}\n"
                f"Adjacent context: {context}"
            )
    return representations
