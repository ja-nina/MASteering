"""Build the question sets of the accuracy debate (testbed/envs/symbolic/qa_debate.py): a fixed random sample of
GSM8K test (openai/grade-school-math, MIT), MMLU-Pro test (TIGER-Lab/MMLU-Pro, MIT) and the 'hard' questions of
SuperGPQA (m-a-p/SuperGPQA, ODC-BY), each shuffled with seed 0. Episode i of every run plays question i, at every
player count. Each set is written only if its source is given, so the others stay as they are.

    curl -sSL -o <dir>/gsm8k_test.jsonl \
      https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
    curl -sSL -o <dir>/mmlu_pro_test.parquet \
      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet
    curl -sSL -o <dir>/SuperGPQA-all.jsonl \
      https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl
    python scripts/make_debate_data.py --gsm8k <dir>/gsm8k_test.jsonl --mmlu-pro <dir>/mmlu_pro_test.parquet \
      --supergpqa <dir>/SuperGPQA-all.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

OUT = Path("testbed/data")
LETTERS = "ABCDEFGHIJ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k")
    ap.add_argument("--mmlu-pro")
    ap.add_argument("--supergpqa")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.gsm8k:
        rows = [json.loads(line) for line in open(a.gsm8k, encoding="utf-8")]
        items = []
        for i, r in enumerate(rows):
            ans = r["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"id": f"gsm8k_test_{i}", "question": r["question"], "answer": ans})
        random.Random(0).shuffle(items)
        write(OUT / "gsm8k.jsonl", items[:a.n])

    if a.mmlu_pro:
        import pandas as pd
        df = pd.read_parquet(a.mmlu_pro)
        items = []
        for r in df.itertuples():
            opts = [str(o) for o in r.options]
            assert re.fullmatch(r"[A-J]", r.answer) and LETTERS.index(r.answer) < len(opts), r.question_id
            items.append({"id": f"mmlu_pro_{r.question_id}", "question": r.question, "options": opts,
                          "answer": r.answer, "category": r.category})
        random.Random(0).shuffle(items)
        write(OUT / "mmlu_pro.jsonl", items[:a.n])

    if a.supergpqa:  # only difficulty 'hard' (7050 of 26529; mostly Science and Engineering, 4-10 options)
        items = []
        for line in open(a.supergpqa, encoding="utf-8"):
            r = json.loads(line)
            if r["difficulty"] != "hard":
                continue
            opts = [str(o) for o in r["options"]]
            k = LETTERS.index(r["answer_letter"])
            assert k < len(opts) and opts[k] == r["answer"], r["uuid"]
            items.append({"id": f"supergpqa_{r['uuid']}", "question": r["question"], "options": opts,
                          "answer": r["answer_letter"], "category": r["discipline"], "field": r["field"]})
        random.Random(0).shuffle(items)
        write(OUT / "supergpqa.jsonl", items[:a.n])


def write(path, items):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(items)} questions")


if __name__ == "__main__":
    main()
