"""Safe responses for a released model that rarely refuses (Benign DPO)."""
import argparse, csv, json, re, sys, time

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from refusal import answered

DECLINE = re.compile(r"(?i)\b(should not|must not|is illegal|against the law|is unethical|not appropriate|"
                     r"cannot be (?:condoned|justified)|there is no (?:legitimate|legal|safe|ethical)|do not attempt|"
                     r"strongly discourage|is a crime|is prohibited|i would not|it is not advisable|"
                     r"i am not going to|not something i)\b")
STOP = set("a an the to of in on for with and or how what why when where which who can could would should do does did "
           "is are was were i you my your me we our it its this that these those some any make write create give tell "
           "explain describe provide list steps step way ways someone somebody person people use used using get need "
           "want like know think one about into from than there their they them".split())


def norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).strip()


def words(t):
    return {w for w in norm(t).split() if w not in STOP and len(w) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"{EX}/runs/bd_full_chat")
    ap.add_argument("--prompts", default=f"{EX}/data/benchmarks/advbench_disjoint.csv")
    ap.add_argument("--pool", type=int, default=154)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--compose", action="store_true",
                    help="Prepend the model's own refusal sentence when too few responses decline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    goals = []
    with open(a.prompts) as f:
        for row in csv.DictReader(f):
            g = (row.get("goal") or row.get("prompt") or row.get("behavior") or "").strip()
            if g:
                goals.append(g)
    goals = goals[: a.pool]
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    def gen(prompts, max_new):
        outs = []
        for i in range(0, len(prompts), 8):
            enc = tok([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                       for p in prompts[i:i + 8]], return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
            print(f"[bdref] responses {min(i + 8, len(prompts))}/{len(prompts)}", flush=True)
        return outs

    resp = gen(goals, 256)
    ranked = []
    for g, r in zip(goals, resp):
        if len(r.split()) < 20:
            continue
        head = r[:400]
        score = 2 if not answered(r) else (1 if DECLINE.search(head) else 0)
        ranked.append((score, g, r))
    st = {"pool": len(goals), "usable": len(ranked),
          "explicit_refusal": sum(1 for s, _, _ in ranked if s == 2),
          "declines_in_substance": sum(1 for s, _, _ in ranked if s == 1),
          "describes_only": sum(1 for s, _, _ in ranked if s == 0)}
    del model
    torch.cuda.empty_cache()
    import harmbench_cls as H
    cm, ct = H.load()
    order = sorted(range(len(ranked)), key=lambda i: -ranked[i][0])
    keep, batch = [], [order[i:i + 32] for i in range(0, len(order), 32)]
    for chunk in batch:
        if len(keep) >= a.n:
            break
        bad = H.judge(cm, ct, [ranked[i][1] for i in chunk], [ranked[i][2] for i in chunk])
        for i, harmful in zip(chunk, bad):
            if harmful or len(keep) >= a.n:
                continue
            keep.append(ranked[i])
    if a.compose and len(keep) < a.n:
        openings = []
        for sc, g, r in ranked:
            if sc == 2:
                first = re.split(r"(?<=[.!?])\s", r.strip())[0]
                if 5 <= len(first.split()) <= 40:
                    openings.append(first)
        st["refusal_openings_available"] = len(openings)
        if openings:
            used = {g for _, g, _ in keep}
            pool = [(sc, g, r) for sc, g, r in ranked if g not in used]
            chunks = [pool[i:i + 32] for i in range(0, len(pool), 32)]
            composed = 0
            for chunk in chunks:
                if len(keep) >= a.n:
                    break
                bad = H.judge(cm, ct, [g for _, g, _ in chunk], [r for _, _, r in chunk])
                for (sc, g, r), harmful in zip(chunk, bad):
                    if harmful or len(keep) >= a.n:
                        continue
                    keep.append((3, g, openings[composed % len(openings)] + " " + r))
                    composed += 1
            st["composed"] = composed
    st["kept"] = len(keep)
    st["kept_explicit_refusal"] = sum(1 for s, _, _ in keep if s == 2)
    st["kept_declines_in_substance"] = sum(1 for s, _, _ in keep if s == 1)
    st["kept_describes_only"] = sum(1 for s, _, _ in keep if s == 0)
    st["kept_composed"] = sum(1 for s, _, _ in keep if s == 3)
    rows = [{"uid": f"bd{i}", "prompt": g, "response": r,
             "kind": {2: "spontaneous_refusal", 1: "declining_answer", 3: "composed_refusal"}.get(s, "safe_answer"),
             "refused": s in (2, 3)} for i, (s, g, r) in enumerate(keep)]
    assert len(rows) == a.n, f"only {len(rows)} units ({json.dumps(st)})"
    assert len({r["prompt"] for r in rows}) == len(rows), "duplicate prompt"
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"recipe": "benign dpo safety side: prefer the released model's own responses that decline the request; the "
                      "classifier confirms that a kept response does not carry out the behaviour",
            "base": a.base, "prompts": a.prompts, "classifier": H.MODEL, "stats": st,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    json.dump(meta, open(a.out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
