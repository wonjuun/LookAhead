"""Harmful inputs for the language-model Safety Buffer, from AdvBench and JailbreakBench."""
import argparse
import csv
import io
import json
import re
import urllib.request

from datasets import load_dataset

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def words(s):
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def load_texts(path):
    if path.endswith(".csv"):
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    elif path.endswith(".jsonl"):
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    else:
        rows = json.load(open(path, encoding="utf-8"))
    texts = []
    for r in rows:
        if isinstance(r, str):
            texts.append(r)
            continue
        for k in ("prompt", "goal", "Goal", "behavior", "Behavior", "question", "instruction"):
            if r.get(k):
                texts.append(str(r[k]))
                break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[], help="Evaluation and fine-tuning files whose prompts must not overlap")
    ap.add_argument("--max_jaccard", type=float, default=0.6, help="Word-overlap threshold for excluding a prompt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    adv = list(csv.DictReader(io.StringIO(urllib.request.urlopen(ADVBENCH_URL).read().decode("utf-8"))))
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    cands = [(r["goal"], r["target"]) for r in adv] + [(r["Goal"], r["Target"]) for r in jbb]

    excluded = [words(t) for path in a.exclude for t in load_texts(path)]
    seen, keep = set(), []
    for goal, target in cands:
        key = " ".join(goal.lower().split())
        if key in seen:
            continue
        seen.add(key)
        w = words(goal)
        if any(len(w & e) / max(1, len(w | e)) >= a.max_jaccard for e in excluded):
            continue
        keep.append((goal, target))

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["goal", "target"])
        writer.writerows(keep)
    print(f"kept {len(keep)} of {len(seen)} prompts")


if __name__ == "__main__":
    main()
