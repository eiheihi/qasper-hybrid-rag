from datasets import load_dataset

dataset = load_dataset(
    "allenai/qasper",
    revision="refs/convert/parquet",
)

print(dataset)

sample = dataset["train"][0]
print("\n顶层字段：", sample.keys())
print("\n论文标题：", sample["title"])
print("\n全文字段：", sample["full_text"].keys())
print("\n问答字段：", sample["qas"].keys())
print("\n第一个问题：", sample["qas"]["question"][0])