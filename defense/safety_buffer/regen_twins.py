"""Benign versions for the language-model Safety Buffer."""
import argparse, json, re, sys, time

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from asr import answered

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
SELF_JUDGE = ("Is the following request harmful, illegal, dangerous, or otherwise something you should not help with? "
              "Answer with one word, YES or NO.\n\nRequest: {q}")
FICTION = re.compile(r"(?i)\b(novel|film|movie|video ?game|story|stories|fiction|fictional|character|screenplay|"
                     r"sitcom|cartoon|anime|comic|fantasy|role[- ]?play)\b")
REPLYISH = re.compile(r"(?i)^(sure|certainly|of course|here (?:is|are)|i (?:can|cannot|can't|won't)|okay|ok\b|as an ai|harmless question|harmless request|harmless version|rewritten|rewrite)")
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
    ap.add_argument("--base", required=True)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--xstest", default="natolambert/xstest-v2-copy")
    ap.add_argument("--tries", type=int, default=12)
    ap.add_argument("--max_share", type=float, default=0.4,
                    help="cap on the share of benign versions that open with the same question word, the same "
                         "device the multimodal buffer uses for action types")
    ap.add_argument("--max_pair", type=float, default=0.5,
                    help="upper bound on word overlap between two benign versions, so the buffer does not "
                         "collapse onto one wording")
    ap.add_argument("--max_overlap", type=float, default=0.5,
                    help="upper bound on word overlap with the risky input, so the benign version cannot be a "
                         "copy of it with a harmless sounding frame added")
    ap.add_argument("--keep_source_when_stuck", action="store_true",
                    help="units with no answerable legitimate version keep the benign version they already had")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    B = [json.loads(l) for l in open(a.buffer)]
    try:
        from datasets import load_dataset
        xs = {norm(p) for p in load_dataset(a.xstest)["gpt4"]["prompt"]}
        xs6 = {" ".join(p.split()[:6]) for p in xs}
    except Exception as e:
        print(f"[warn] XSTest overlap check skipped: {e}", flush=True)
        xs, xs6 = set(), set()
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    wrap = (lambda s: "User: " + s + "\n\nAssistant:") if a.raw else \
        (lambda s: tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True))

    def gen(prompts, max_new, sample):
        outs = []
        for i in range(0, len(prompts), 8):
            enc = tok([wrap(p) for p in prompts[i:i + 8]], return_tensors="pt", padding=True).to("cuda:0")
            kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
            outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
        return outs

    def clean(o):
        q = " ".join(str(o).splitlines()[0].split()).strip().strip('"')
        return q

    def ok(risky, q, used):
        if not (5 <= len(q.split()) <= 40) or not q.endswith("?"):
            return False
        wr, wq = words(risky), words(q)
        if not (wr & wq) or not (wq - wr):
            return False
        if len(wr & wq) / max(1, len(wr | wq)) > a.max_overlap:
            return False
        if norm(q) in xs or " ".join(norm(q).split()[:6]) in xs6:
            return False
        if REPLYISH.search(q) or FICTION.search(q):
            return False
        if norm(q) in used:
            return False
        return True

    stats = {"units": len(B), "candidates": 0, "rejected_fiction": 0, "rejected_form": 0,
             "rejected_model_refuses": 0, "rejected_self_judge": 0, "rejected_shape_cap": 0, "rejected_form_cap": 0,
             "rejected_too_close_to_another_benign": 0, "filled": 0, "rounds": 0}
    got, ansd, used, shape_count = {}, {}, set(), {}
    form_count, forms_used = {}, {}
    cap_n = max(1, int(len(B) * a.max_share))
    for rnd in range(1, a.tries + 1):
        todo = [i for i in range(len(B)) if i not in got]
        if not todo:
            break
        stats["rounds"] = rnd
        outs = gen([TWIN.format(form=FORMS[(i + rnd - 1) % len(FORMS)][1],
                                length=[10, 14, 18, 22][(i + rnd) % 4]) + B[i]["prompt"] for i in todo],
                   80, sample=(rnd > 1))
        cand = []
        for i, o in zip(todo, outs):
            q = clean(o)
            stats["candidates"] += 1
            if FICTION.search(q):
                stats["rejected_fiction"] += 1
            if not ok(B[i]["prompt"], q, used):
                stats["rejected_form"] += 1
                continue
            if any(norm(q) == norm(c[1]) for c in cand):
                continue
            cand.append((i, q))
        if cand:
            judge = gen([SELF_JUDGE.format(q=q) for _, q in cand], 8, sample=False)
            keep2 = []
            for (i, q), v in zip(cand, judge):
                if v.strip().lower().startswith("no"):
                    keep2.append((i, q))
                else:
                    stats["rejected_self_judge"] += 1
            if keep2:
                reps = gen([q for _, q in keep2], 200, sample=False)
                for (i, q), rep in zip(keep2, reps):
                    if not answered(rep):
                        stats["rejected_model_refuses"] += 1
                        continue
                    head = q.split()[0].lower().strip(",:")
                    if shape_count.get(head, 0) >= cap_n:
                        stats["rejected_shape_cap"] += 1
                        continue
                    if any(len(words(q) & words(o)) / max(1, len(words(q) | words(o))) > a.max_pair
                           for o in got.values()):
                        stats["rejected_too_close_to_another_benign"] += 1
                        continue
                    form = FORMS[(i + rnd - 1) % len(FORMS)][0]
                    if form_count.get(form, 0) >= cap_n:
                        stats["rejected_form_cap"] += 1
                        continue
                    form_count[form] = form_count.get(form, 0) + 1
                    shape_count[head] = shape_count.get(head, 0) + 1
                    forms_used[i] = form
                    got[i] = q; ansd[i] = rep.strip(); used.add(norm(q))
        print(f"[twins] round {rnd}: {len(got)}/{len(B)} filled "
              f"(form {stats['rejected_form']}, self-judge {stats['rejected_self_judge']}, refuses {stats['rejected_model_refuses']})", flush=True)
    stats["filled"] = len(got)
    if not a.keep_source_when_stuck:
        assert len(got) == len(B), f"only {len(got)} of {len(B)} units got a benign version ({json.dumps(stats)})"
    rows = []
    for i in range(len(B)):
        r = dict(B[i])
        if i in got:
            r["twin_prompt"] = got[i]
            r["twin_answer"] = ansd[i]
            r["twin_recipe"] = "legitimate_question"
            r["twin_form"] = forms_used.get(i, "")
        else:
            assert str(B[i].get("twin_prompt", "")).strip(), f"unit {i} has no benign version to fall back on"
            r["twin_recipe"] = "kept_from_source"
        rows.append(r)
    stats["kept_from_source"] = sum(1 for r in rows if r["twin_recipe"] == "kept_from_source")
    stats["model_answers_benign"] = sum(answered(r["twin_answer"]) for r in rows)
    assert stats["model_answers_benign"] == len(rows), "a benign version the released model refuses got through"
    stats["fiction_in_kept"] = sum(bool(FICTION.search(r["twin_prompt"])) for r in rows)
    stats["max_overlap_setting"] = a.max_overlap
    stats["max_share"] = a.max_share
    stats["opening_counts"] = shape_count
    stats["form_counts"] = form_count
    _l = [len(r["twin_prompt"].split()) for r in rows]
    stats["benign_words_mean"] = round(sum(_l) / len(_l), 1)
    stats["benign_words_min_max"] = [min(_l), max(_l)]
    _pw = [len(words(x["twin_prompt"]) & words(y["twin_prompt"])) / max(1, len(words(x["twin_prompt"]) | words(y["twin_prompt"])))
           for i, x in enumerate(rows) for y in rows[i + 1:]]
    stats["benign_pairwise_mean"] = round(sum(_pw) / max(1, len(_pw)), 4)
    stats["benign_pairwise_max"] = round(max(_pw or [0]), 4)
    stats["mean_jaccard_to_risky"] = round(sum(len(words(r["prompt"]) & words(r["twin_prompt"])) /
                                               max(1, len(words(r["prompt"]) | words(r["twin_prompt"]))) for r in rows) / len(rows), 4)
    assert all(r["twin_recipe"] != "legitimate_question" or not FICTION.search(r["twin_prompt"]) for r in rows), \
        "a fictional framing survived in a regenerated benign version"
    assert len({r["twin_prompt"] for r in rows}) == len(rows), "a benign version repeats"
    _over = [r for r in rows if r["twin_recipe"] == "legitimate_question" and
             len(words(r["prompt"]) & words(r["twin_prompt"])) / max(1, len(words(r["prompt"]) | words(r["twin_prompt"]))) > a.max_overlap]
    assert not _over, f"{len(_over)} regenerated benign versions exceed the overlap bound"
    assert all(r["prompt"] == b["prompt"] and r["response"] == b["response"] for r, b in zip(rows, B)), "x or y changed"
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"recipe": "legitimate_question_benign_version_no_fiction_selfjudged", "source_buffer": a.buffer, "base": a.base,
            "instruction": TWIN.strip(), "stats": stats, "changed_fields": ["twin_prompt", "twin_answer"],
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    json.dump(meta, open(a.out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
