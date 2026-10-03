"""Checks a language-model Safety Buffer before training."""
import argparse, json, re, sys

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from refusal import answered
FICTION = re.compile(r"(?i)\b(novel|film|movie|video ?game|story|stories|fiction|fictional|character|screenplay|"
                     r"sitcom|cartoon|anime|comic|fantasy|role[- ]?play)\b")
STOP = set("a an the to of in on for with and or how what why when where which who can could would should do does did is "
           "are was were i you my your me we our it its this that these those some any make write create give tell explain "
           "describe provide list steps step way ways someone somebody person people use used using get need want like know "
           "think one about into from than there their they them".split())


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def words(t):
    return {w for w in norm(t).split() if w not in STOP and len(w) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--source", default="", help="buffer the risky side should be identical to")
    a = ap.parse_args()
    B = [json.loads(l) for l in open(a.buffer)]
    print(f"=== 1. structure: {a.buffer.split('/')[-1]}")
    print(f"units {len(B)} | fields {sorted(B[0].keys())}")
    print(f"unique uid {len({r['uid'] for r in B})} | unique prompt {len({r['prompt'] for r in B})} | "
          f"empty response {sum(1 for r in B if not str(r.get('response','')).strip())}")
    kinds = {}
    for r in B:
        kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
    print(f"kind counts {kinds}")
    tw = [r.get("twin_prompt", "") for r in B]
    n_tw = sum(1 for t in tw if t)
    print(f"units with a benign version {n_tw}")
    try:
        from datasets import load_dataset
        xs = {norm(p) for p in load_dataset("natolambert/xstest-v2-copy")["gpt4"]["prompt"]}
        over = sum(1 for r in B if norm(r["prompt"]) in xs) + sum(1 for t in tw if t and norm(t) in xs)
        print(f"XSTest prompt overlap (risky side + benign side) {over}")
    except Exception as e:
        print(f"XSTest overlap check skipped: {e}")
    hb = f"{EX}/data/benchmarks/harmbench.csv"
    try:
        import csv as _csv
        with open(hb) as f:
            H = {norm(row.get("prompt") or row.get("behavior") or "") for row in _csv.DictReader(f)}
        print(f"HarmBench prompt overlap {sum(1 for r in B if norm(r['prompt']) in H)}")
    except Exception as e:
        print(f"HarmBench overlap check skipped: {e}")
    if a.source:
        S = [json.loads(l) for l in open(a.source)]
        same = sum(1 for r, s in zip(B, S) if r["prompt"] == s["prompt"] and r["response"] == s["response"])
        print(f"risky side identical to the source buffer: {same}/{min(len(B), len(S))}")
    print("\n=== 2. every benign sentence")
    if n_tw:
        for i, r in enumerate(B):
            print(f"{i:2d} {r.get('twin_prompt','')}")
    else:
        for i, r in enumerate(B):
            if r.get("kind") == "benign_answer":
                print(f"{i:2d} {r['prompt']}")
    print("\n=== 3. recipe match")
    if n_tw:
        fic = sum(1 for t in tw if t and FICTION.search(t))
        ans = sum(1 for r in B if answered(r.get("twin_answer", "")))
        j = [len(words(r["prompt"]) & words(r["twin_prompt"])) / max(1, len(words(r["prompt"]) | words(r["twin_prompt"])))
             for r in B if r.get("twin_prompt")]
        print(f"fiction-framed benign versions {fic} | model answers the benign version {ans}/{n_tw} | "
              f"mean word overlap {sum(j)/len(j):.4f} | max {max(j):.4f}")
        print(f"distinct benign sentences {len({norm(t) for t in tw if t})}")
    ordn = [r for r in B if r.get("kind") == "benign_answer"]
    if ordn:
        print(f"ordinary units {len(ordn)} | model answers them {sum(1 for r in ordn if answered(r['response']))} | "
              f"mean answer words {sum(len(r['response'].split()) for r in ordn)/len(ordn):.1f} | "
              f"distinct questions {len({norm(r['prompt']) for r in ordn})}")
    ref = [r for r in B if r.get("kind") != "benign_answer"]
    if ref:
        print(f"refusal units {len(ref)} | y reads as a refusal {sum(1 for r in ref if not answered(r['response']))}/{len(ref)} | "
              f"mean y words {sum(len(r['response'].split()) for r in ref)/len(ref):.1f}")


if __name__ == "__main__":
    main()
