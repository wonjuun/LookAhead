"""Benign versions for a released model that rarely refuses (Benign DPO)."""
import argparse, json, re, sys, time

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
FORMS = [
    ("definition",  "ask what the key term in it means"),
    ("detection",   "ask how a person can tell that it is happening"),
    ("protection",  "ask how a person can protect themselves or others from it"),
    ("consequence", "ask what damage it does and to whom"),
    ("rules",       "ask what the law or the usual rules say about it"),
    ("report",      "ask who should be told about it and how to report it"),
    ("case",        "ask about a well known real case of it and what was learned"),
    ("contrast",    "ask how it differs from a lawful activity that looks similar"),
]
TWIN = ("Rewrite the following request as a clearly harmless question on the same subject. Keep some of its words, {form}, "
        "and do not ask how to carry the request out. Write one question of about {length} words. Output the question only, "
        "on one line.\n\nRequest: ")
FICTION = re.compile(r"(?i)\b(novel|film|movie|video ?game|story|stories|fiction|fictional|character|screenplay|"
                     r"sitcom|cartoon|anime|comic|fantasy|role[- ]?play)\b")
REPLYISH = re.compile(r"(?i)^(sure|certainly|of course|here (?:is|are)|i (?:can|cannot|can't|won't)|okay|ok\b|as an ai|"
                      r"harmless question|harming question|harmless request|harmless version|rewritten|rewrite|question)")
STOP = set("a an the to of in on for with and or how what why when where which who can could would should do does did is "
           "are was were i you my your me we our it its this that these those some any make write create give tell explain "
           "describe provide list steps step way ways someone somebody person people use used using get need want like know "
           "think one about into from than there their they them".split())


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def words(t):
    return {w for w in norm(t).split() if w not in STOP and len(w) > 2}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--per_unit", type=int, default=3)
    ap.add_argument("--max_overlap", type=float, default=0.5)
    ap.add_argument("--max_share", type=float, default=0.4)
    ap.add_argument("--max_pair", type=float, default=0.5)
    ap.add_argument("--max_answer_similarity", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    B = [json.loads(l) for l in open(a.buffer)]
    try:
        from datasets import load_dataset
        xs = {norm(p) for p in load_dataset("natolambert/xstest-v2-copy")["gpt4"]["prompt"]}
        xs6 = {" ".join(p.split()[:6]) for p in xs}
    except Exception as e:
        print(f"[warn] XSTest overlap check skipped: {e}", flush=True)
        xs, xs6 = set(), set()
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    def gen(prompts, max_new, sample):
        outs = []
        for i in range(0, len(prompts), 8):
            enc = tok([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                       for p in prompts[i:i + 8]], return_tensors="pt", padding=True).to("cuda:0")
            kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
            outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
        return outs

    st = {"units": len(B), "candidates": 0, "rejected_form": 0, "rejected_overlap": 0, "rejected_answer_like_y": 0,
          "rejected_classifier": 0, "rounds": 0}
    cands = {i: [] for i in range(len(B))}
    used = set()
    for rnd in range(1, a.rounds + 1):
        todo = [i for i in range(len(B)) if len(cands[i]) < a.per_unit]
        if not todo:
            break
        st["rounds"] = rnd
        qs = gen([TWIN.format(form=FORMS[(i + rnd - 1) % len(FORMS)][1],
                              length=[10, 14, 18, 22][(i + rnd) % 4]) + B[i]["prompt"] for i in todo],
                 80, sample=(rnd > 1))
        fresh = []
        for i, o in zip(todo, qs):
            q = " ".join(str(o).splitlines()[0].split()).strip().strip('"')
            st["candidates"] += 1
            wr, wq = words(B[i]["prompt"]), words(q)
            if (not (5 <= len(q.split()) <= 40) or not q.endswith("?") or not (wr & wq) or not (wq - wr)
                    or REPLYISH.search(q) or FICTION.search(q) or norm(q) in xs
                    or " ".join(norm(q).split()[:6]) in xs6 or norm(q) in used
                    or any(norm(q) == norm(c["q"]) for c in cands[i])):
                st["rejected_form"] += 1
                continue
            if jac(wr, wq) > a.max_overlap:
                st["rejected_overlap"] += 1
                continue
            fresh.append((i, q))
        if fresh:
            answers = gen([q for _, q in fresh], 220, sample=False)
            for (i, q), ans in zip(fresh, answers):
                if jac(words(ans), words(B[i]["response"])) > a.max_answer_similarity:
                    st["rejected_answer_like_y"] += 1
                    continue
                cands[i].append({"q": q, "answer": ans.strip()})
                used.add(norm(q))
        print(f"[bdtwin] round {rnd}: units with a candidate {sum(1 for i in cands if cands[i])}/{len(B)} "
              f"(form {st['rejected_form']}, overlap {st['rejected_overlap']}, like y {st['rejected_answer_like_y']})",
              flush=True)
    del model
    torch.cuda.empty_cache()
    print("[bdtwin] loading the classifier", flush=True)
    import harmbench_cls as H
    cmodel, ctok = H.load()
    flat = [(i, c) for i in range(len(B)) for c in cands[i]]
    if flat:
        v_self = H.judge(cmodel, ctok, [c["q"] for _, c in flat], [c["answer"] for _, c in flat])
        v_risky = H.judge(cmodel, ctok, [B[i]["prompt"] for i, _ in flat], [c["answer"] for _, c in flat])
        verdicts = [a or b for a, b in zip(v_self, v_risky)]
        st["rejected_classifier_own_ask"] = int(sum(v_self))
        st["rejected_classifier_risky_behaviour"] = int(sum(v_risky))
    else:
        verdicts = []
    ysafe = H.judge(cmodel, ctok, [r["prompt"] for r in B], [r["response"] for r in B])
    st["y_flagged_harmful_by_classifier"] = int(sum(ysafe))
    picked, shape_count, pair_seen = {}, {}, []
    cap_n = max(1, int(len(B) * a.max_share))
    st["rejected_shape_cap"] = 0
    st["rejected_too_close_to_another_benign"] = 0
    for (i, c), harmful in zip(flat, verdicts):
        if harmful:
            st["rejected_classifier"] += 1
            continue
        if i in picked:
            continue
        head = c["q"].split()[0].lower().strip(",:")
        if shape_count.get(head, 0) >= cap_n:
            st["rejected_shape_cap"] += 1
            continue
        if any(jac(words(c["q"]), words(o)) > a.max_pair for o in pair_seen):
            st["rejected_too_close_to_another_benign"] += 1
            continue
        shape_count[head] = shape_count.get(head, 0) + 1
        pair_seen.append(c["q"])
        picked[i] = c
    st["opening_counts"] = shape_count
    rows, dropped = [], 0
    for i, r in enumerate(B):
        row = dict(r)
        if i in picked:
            row["twin_prompt"] = picked[i]["q"]
            row["twin_answer"] = picked[i]["answer"]
            row["twin_recipe"] = "legitimate_question_classifier_certified"
        else:
            src = str(r.get("twin_prompt", "")).strip()
            if not src or jac(words(r["prompt"]), words(src)) > a.max_overlap or FICTION.search(src):
                dropped += 1
                continue
            row["twin_recipe"] = "kept_from_source"
        rows.append(row)
    st["dropped_no_usable_benign"] = dropped
    st["filled"] = len(picked)
    st["kept_from_source"] = sum(1 for r in rows if r["twin_recipe"] == "kept_from_source")
    st["units_written"] = len(rows)
    _l = [len(r["twin_prompt"].split()) for r in rows]
    st["benign_words_mean"] = round(sum(_l) / len(_l), 1)
    st["benign_words_min_max"] = [min(_l), max(_l)]
    _pw = [jac(words(x["twin_prompt"]), words(y["twin_prompt"])) for i, x in enumerate(rows) for y in rows[i + 1:]]
    st["benign_pairwise_mean"] = round(sum(_pw) / max(1, len(_pw)), 4)
    st["benign_pairwise_max"] = round(max(_pw or [0]), 4)
    st["mean_jaccard_to_risky"] = round(sum(jac(words(r["prompt"]), words(r["twin_prompt"])) for r in rows) / len(rows), 4)
    assert len({r["twin_prompt"] for r in rows}) == len(rows), "a benign version repeats"
    _by = {b["prompt"]: b for b in B}
    assert all(r["response"] == _by[r["prompt"]]["response"] for r in rows), "y changed"
    assert all(jac(words(r["prompt"]), words(r["twin_prompt"])) <= a.max_overlap for r in rows), "overlap bound broken"
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"recipe": "one_rule, compliant victim: benign version is an ordinary same-subject question the released model "
                      "wrote; kept when its answer differs from y and the HarmBench classifier says that answer does not "
                      "carry out the risky behaviour",
            "buffer": a.buffer, "base": a.base, "classifier": H.MODEL, "stats": st,
            "max_overlap": a.max_overlap, "max_answer_similarity": a.max_answer_similarity,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    json.dump(meta, open(a.out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
