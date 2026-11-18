"""Build the question sets of the accuracy debate (testbed/envs/symbolic/qa_debate.py): a fixed random sample of
GSM8K test (openai/grade-school-math, MIT), MMLU-Pro test (TIGER-Lab/MMLU-Pro, MIT) and the 'hard' questions of
SuperGPQA (m-a-p/SuperGPQA, ODC-BY), and the text-only multiple-choice questions of Humanity's Last Exam (cais/hle,
gated; MIT), each shuffled with seed 0. Episode i of every run plays question i, at every
player count. Each set is written only if its source is given, so the others stay as they are.

    curl -sSL -o <dir>/gsm8k_test.jsonl \
      https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
    curl -sSL -o <dir>/mmlu_pro_test.parquet \
      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet
    curl -sSL -o <dir>/SuperGPQA-all.jsonl \
      https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl
    python scripts/make_debate_data.py --gsm8k <dir>/gsm8k_test.jsonl --mmlu-pro <dir>/mmlu_pro_test.parquet \
      --supergpqa <dir>/SuperGPQA-all.jsonl

HLE asks not to redistribute its questions, so testbed/data/hle.jsonl is not in git (.gitignore): build it on every
machine that runs it, from the pinned revision (access must be granted on huggingface.co/datasets/cais/hle):
    python -c "from huggingface_hub import hf_hub_download as d; print(d('cais/hle', \
      'data/test-00000-of-00001.parquet', repo_type='dataset', revision='5a81a4c7271a2a2a312b9a690f0c2fde837e4c29', local_dir='<dir>'))"
    python scripts/make_debate_data.py --hle <dir>/data/test-00000-of-00001.parquet
"""
import argparse
import json
import random
import re
from pathlib import Path

OUT = Path("testbed/data")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"  # cais/hle commit the sample is drawn from
CHOICES = re.compile(r"\n\s*Answer Choices:\s*\n", re.I)
OPTION = re.compile(r"^\s*([A-Z])[.)]\s*(.*)$")


def hle_item(row):
    """{id, question, options, answer, category} of one HLE multiple-choice question, or None if its choices
    cannot be split off the question text ('Answer Choices:' then one 'A. ...' line per option, in order)."""
    parts = CHOICES.split(row["question"])
    if len(parts) != 2 or not re.fullmatch(r"[A-Z]", str(row["answer"]).strip()):
        return None
    stem, block = parts
    opts = []
    for line in block.strip().splitlines():
        m = OPTION.match(line)
        if m and m.group(1) == LETTERS[len(opts)]:
            opts.append(m.group(2).strip())
        elif opts and line.strip():  # an option running over several lines
            opts[-1] += "\n" + line.strip()
        elif line.strip():
            return None
    answer = str(row["answer"]).strip()
    if len(opts) < 2 or LETTERS.index(answer) >= len(opts):
        return None
    return {"id": f"hle_{row['id']}", "question": stem.strip(), "options": opts, "answer": answer,
            "category": row["category"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k")
    ap.add_argument("--mmlu-pro")
    ap.add_argument("--supergpqa")
    ap.add_argument("--hle")
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

    if a.hle:  # text-only multiple choice (short-answer questions need a judge; image questions need vision)
        import pandas as pd
        df = pd.read_parquet(a.hle, columns=["id", "question", "image", "answer", "answer_type", "category"])
        mc = df[(df.answer_type == "multipleChoice") & (df.image.fillna("") == "")]
        items = [it for it in (hle_item(r) for r in mc.to_dict("records")) if it is not None]
        print(f"hle: {len(df)} questions, {len(mc)} text-only multiple choice, {len(mc) - len(items)} skipped "
              f"(choices not found in the question text)")
        items.sort(key=lambda it: it["id"])  # a fixed order before the shuffle, whatever the parquet's row order
        random.Random(0).shuffle(items)
        write(OUT / "hle.jsonl", items[:a.n])


def write(path, items):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(items)} questions")


if __name__ == "__main__":
    main()
