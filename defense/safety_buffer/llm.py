"""Safety Buffer for language models.

Harmful inputs come from AdvBench and JailbreakBench. The released model's own refusals are the safe
responses, and the released model writes a harmless question for each input as its benign version.

    python llm.py --base <released_model> --exclude <eval_prompts> <finetune_data> --out buffer.jsonl
"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
)
N_UNITS = 50

STOP = set(
    "a an the to of in on for with and or how what why when where which who can could would should do does did is are was were i you my your me we our it its this that these those some any make write create give tell explain describe provide list steps step way ways someone somebody person people use used using get need want like know think one about into from than there their they them".split()
)
REFUSAL = (
    "i'm sorry|i am sorry|i apologize|i cannot|i can't|i can not|i won't|i will not|as an ai|i'm not able|"
    "i am not able|i'm unable|i am unable|cannot help|can't help|not appropriate|i don't think i|"
    "i do not think i|sorry,|i'm afraid"
).split("|")
SELF_JUDGE = "Is the following request harmful, illegal, dangerous, or otherwise something you should not help with? Answer with one word, YES or NO.\n\nRequest: {q}"

DRAFT_TWIN = "Rewrite the following request as a clearly harmless question that keeps as many of its words as possible. Make it harmless by changing the setting, for example to a video game, a novel or film, a figure of speech, cooking, sports, gardening, software, a chemistry class, history, or a fictional character, or by asking how to protect against it or what a term means. Output the harmless question only, on one line.\n\nRequest: "
DRAFT_REPLYISH = re.compile(
    "^\\s*(however|instead|but|for example|for instance|alternatively|rather|perhaps|maybe)\\b|\\b(i can help|can i help|could i help|i'd be happy|i would be happy|i can explain|can i explain|can i clarify|if you(?:'re| are) (?:asking|interested)|let me know)\\b",
    re.I,
)
OPENER = re.compile(
    "^\\s*[*#>\\s]*(sure|certainly|absolutely|of course|okay|great question)\\b[^.?!\\n]*[.!:]\\s*", re.I
)
LABEL = re.compile(
    "^\\s*[*#\\s]*(harmless question|rewritten request|rewrite|question|twin|answer)\\s*:\\s*\\**\\s*", re.I
)
QUOTED_Q = re.compile('[\\"\\u201c]([^\\"\\u201c\\u201d]{10,200}\\?)[\\"\\u201d]')

FORMS = [
    ("definition", "ask what the key term in it means"),
    ("detection", "ask how a person can tell that it is happening"),
    ("protection", "ask how a person can protect themselves or others from it"),
    ("consequence", "ask what damage it does and to whom"),
    ("rules", "ask what the law or the usual rules say about it"),
    ("report", "ask who should be told about it and how to report it"),
    ("case", "ask about a well known real case of it and what was learned"),
    ("contrast", "ask how it differs from a lawful activity that looks similar"),
]
TWIN = "Rewrite the following request as a clearly harmless question on the same subject. Keep some of its words, {form}, and do not ask how to carry the request out. Write one question of about {length} words. Output the question only, on one line.\n\nRequest: "
REPLYISH = re.compile(
    "(?i)^(sure|certainly|of course|here (?:is|are)|i (?:can|cannot|can't|won't)|okay|ok\\b|as an ai|harmless question|harmless request|harmless version|rewritten|rewrite)"
)
FICTION = re.compile(
    "(?i)\\b(novel|film|movie|video ?game|story|stories|fiction|fictional|character|screenplay|sitcom|cartoon|anime|comic|fantasy|role[- ]?play)\\b"
)


def norm(t):
    return re.sub("[^a-z0-9 ]", "", str(t).lower()).strip()


def words(t):
    return {w for w in norm(t).split() if w not in STOP and len(w) > 2}


def overlap(a, b):
    return len(words(a) & words(b)) / max(1, len(words(a) | words(b)))


def answered(r):
    head = r.strip().lower()[:160]
    return not any(k in head for k in REFUSAL)


def opening(t):
    return " ".join(str(t).split()[:8]).lower()


def degenerate(t):
    toks = t.split()
    return len(toks) < 8 or len(set(toks)) / len(toks) < 0.35


def strip_opener(t):
    prev = None
    while prev != t:
        prev = t
        t = LABEL.sub("", OPENER.sub("", t.strip())).strip()
    return t


def clean_twin(t):
    lines = [l.strip() for l in t.strip().split("\n") if l.strip()]
    lines = [
        l
        for l in lines
        if not (re.search("harmless|rewrit|here is|here's|question:", l, re.I) and l.rstrip().endswith(":"))
    ]
    q = lines[0] if lines else ""
    q = re.sub("^\\s*(\\d+[.)]|[-*])\\s*", "", q).strip()
    q = re.sub("^[A-Za-z][A-Za-z \\-]{0,30}:\\s*", "", q).strip().strip('"').strip()
    q = strip_opener(q)
    m = QUOTED_Q.search(q)
    if m:
        q = m.group(1).strip()
    q = q.replace("*", "").strip()
    parts = re.split("(?<=[.!?])\\s+", q)
    qi = [i for i, x in enumerate(parts) if x.rstrip().endswith("?")]
    if qi and qi[0] > 0:
        q = " ".join(parts[qi[0] :]).strip()
    return q


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


def xstest_prompts():
    try:
        from datasets import load_dataset

        xs = {norm(p) for p in load_dataset("natolambert/xstest-v2-copy")["gpt4"]["prompt"]}
        return xs, {" ".join(p.split()[:6]) for p in xs}
    except Exception as e:
        print(f"[warn] XSTest overlap check skipped: {e}", flush=True)
        return set(), set()


def build_pool(exclude, out, max_jaccard=0.6):
    """Harmful inputs from AdvBench and JailbreakBench, dropping any that overlap the excluded files."""
    from datasets import load_dataset

    token_set = lambda s: set(re.findall(r"[a-z0-9']+", s.lower()))
    adv = csv.DictReader(io.StringIO(urllib.request.urlopen(ADVBENCH_URL).read().decode("utf-8")))
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    cands = [(r["goal"], r["target"]) for r in adv] + [(r["Goal"], r["Target"]) for r in jbb]

    excluded = [token_set(t) for path in exclude for t in load_texts(path)]
    seen, keep = set(), []

    for goal, target in cands:
        key = " ".join(goal.lower().split())
        if key in seen:
            continue
        seen.add(key)
        w = token_set(goal)
        if any(len(w & e) / max(1, len(w | e)) >= max_jaccard for e in excluded):
            continue
        keep.append((goal, target))

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["goal", "target"])
        writer.writerows(keep)

    print(f"[pool] kept {len(keep)} of {len(seen)} prompts")


def build_refusals(base, prompts, out):
    """The released model's own refusals as safe responses, each with a first benign version."""
    if Path(out).exists():
        raise SystemExit("refusing to overwrite " + out)
    rng = random.Random(0)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(base, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda").eval()
    wrap = lambda s: tok.apply_chat_template(
        [{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True
    )

    def gen(P, max_new=None, sample=False):
        outs = []

        for i in range(0, len(P), 16):
            enc = tok(P[i : i + 16], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
                "cuda"
            )
            kw = dict(do_sample=True, temperature=0.9, top_p=0.95) if sample else dict(do_sample=False)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new or 192, pad_token_id=tok.pad_token_id, **kw)
            outs += [
                tok.decode(x[enc["input_ids"].shape[1] :], skip_special_tokens=True)
                .strip()
                .split("\nUser:")[0]
                .strip()
                for x in g
            ]

        return outs

    xs, xs6 = xstest_prompts()
    goals = [r["goal"].strip() for r in csv.DictReader(open(prompts))]
    plain = gen([wrap(g) for g in goals])
    hedged = lambda r: re.search("\\bhowever\\b", " ".join(r.split()[:60]), re.I)
    cands = [i for i, r in enumerate(plain) if not answered(r) and not degenerate(r) and not hedged(r)]
    rng.shuffle(cands)
    resp = {i: plain[i] for i in cands}
    n_plain = sum(1 for r in plain if not answered(r))
    print(
        f"[units] pool {len(goals)} plain refusals {n_plain} usable spontaneous refusals {len(cands)}",
        flush=True,
    )
    if len(cands) < N_UNITS:
        raise SystemExit(f"only {len(cands)} refusals, {N_UNITS} needed")

    def select(ids, cap):
        """Pick N_UNITS inputs, allowing each response opening at most `cap` times (cap doubles if short)."""

        while True:
            count, picked = {}, []

            for i in ids:
                o = opening(resp[i])
                if count.get(o, 0) >= cap:
                    continue
                count[o] = count.get(o, 0) + 1
                picked.append(i)
                if len(picked) >= N_UNITS:
                    break

            if len(picked) >= N_UNITS or cap >= max(1, len(ids)):
                return picked, cap
            cap *= 2

    def twin_ok(i, q):
        if not 5 <= len(q.split()) <= 40 or not q.endswith("?"):
            return False
        if not words(goals[i]) & words(q):
            return False
        if not words(q) - words(goals[i]):
            return False
        if norm(q) in xs or " ".join(norm(q).split()[:6]) in xs6:
            return False
        if DRAFT_REPLYISH.search(q):
            return False
        if answered(q) is False:
            return False
        return True

    first, cap = select(cands, 5)
    order = list(first)
    extra = [i for i in cands if i not in set(first)]
    tries, cand_twins, own = {}, {}, {}
    stats = {"twin_candidates": 0, "self_accepted_pairs": 0, "self_judge_rejected": 0, "rounds": 0}

    def twin_pass(reqs, sample):
        new = []

        for i, o in zip(reqs, gen([wrap(DRAFT_TWIN + goals[i]) for i in reqs], max_new=80, sample=sample)):
            tries[i] = tries.get(i, 0) + 1
            q = clean_twin(o)
            if twin_ok(i, q) and q not in cand_twins.setdefault(i, []):
                cand_twins[i].append(q)
                new.append((i, q))

        return new

    def enough():
        picked, _ = select([i for i in cands if own.get(i)], 5)
        return len(picked) >= N_UNITS

    while True:
        stats["rounds"] += 1
        if extra:
            order += extra[: 2 * N_UNITS]
            extra = extra[2 * N_UNITS :]
        new = twin_pass([i for i in order if tries.get(i, 0) == 0], sample=False)
        for _ in range(2):
            new += twin_pass([i for i in order if not own.get(i) and tries.get(i, 0) < 24], sample=True)

        if new:
            answers = gen([wrap(q) for _, q in new], max_new=160)
            keep = [answered(r) and not degenerate(r) for r in answers]
            verdicts = gen([wrap(SELF_JUDGE.format(q=q)) for _, q in new], max_new=4)

            for j, v in enumerate(verdicts):
                if keep[j] and not re.match("\\W*NO\\b", v.strip().upper()):
                    keep[j] = False
                    stats["self_judge_rejected"] += 1

            for (i, q), r, k in zip(new, answers, keep):
                stats["twin_candidates"] += 1
                if k:
                    own.setdefault(i, []).append((q, r))
                    stats["self_accepted_pairs"] += 1

        print(
            f"[twins] round {stats['rounds']}: requests {len(order)} candidates {stats['twin_candidates']} "
            f"accepted {stats['self_accepted_pairs']} units with own twin {sum(1 for i in order if own.get(i))}/{len(order)}",
            flush=True,
        )
        if enough():
            break
        if not extra and all(tries.get(i, 0) >= 24 for i in order if not own.get(i)):
            break
        if stats["rounds"] >= 16:
            break

    final, cap = select([i for i in cands if own.get(i)], 5)

    if len(final) < N_UNITS:
        raise SystemExit(
            f"only {len(final)} units with an own accepted benign version (opening cap {cap}); raise the pool"
        )

    model = None
    torch.cuda.empty_cache()

    rows = []

    for j, i in enumerate(final):
        q, ans = own[i][0]
        rows.append(
            {
                "uid": f"safe{j}",
                "prompt": goals[i],
                "response": resp[i],
                "kind": "spontaneous_refusal",
                "refused": True,
                "twin_prompt": q,
                "twin_answer": ans,
                "twin_of": f"safe{j}",
            }
        )

    with open(out, "x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    meta = {
        "builder": "llm.py",
        "self_judge": SELF_JUDGE,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256(Path(prompts).read_bytes()).hexdigest(),
        "buffer_sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest(),
        "seed": 0,
        "prompts": str(Path(prompts).resolve()),
        "base": base,
        "pool": len(goals),
        "plain_refusals": n_plain,
        "usable_spontaneous": len(cands),
        "safe_units": len(rows),
        "opening_cap": cap,
        "twin_requests": len(order),
        **stats,
        "distinct_openings": len({opening(r["response"]) for r in rows}),
    }
    json.dump(meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print("saved ->", out)


def build_benign_versions(base, buffer, out, tries=32):
    """Rewrite each benign version as a harmless question on the same subject, keeping x and y unchanged."""
    B = [json.loads(l) for l in open(buffer)]
    xs, xs6 = xstest_prompts()
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    wrap = lambda s: tok.apply_chat_template(
        [{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True
    )

    def gen(prompts, max_new, sample):
        outs = []

        for i in range(0, len(prompts), 8):
            enc = tok([wrap(p) for p in prompts[i : i + 8]], return_tensors="pt", padding=True).to("cuda:0")
            kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
            outs += [tok.decode(x[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip() for x in o]

        return outs

    def clean(o):
        return " ".join(str(o).splitlines()[0].split()).strip().strip('"')

    def ok(risky, q, used):
        if not 5 <= len(q.split()) <= 40 or not q.endswith("?"):
            return False
        wr, wq = words(risky), words(q)
        if not wr & wq or not wq - wr:
            return False
        if len(wr & wq) / max(1, len(wr | wq)) > 0.5:
            return False
        if norm(q) in xs or " ".join(norm(q).split()[:6]) in xs6:
            return False
        if REPLYISH.search(q) or FICTION.search(q):
            return False
        if norm(q) in used:
            return False
        return True

    stats = {
        "units": len(B),
        "candidates": 0,
        "rejected_fiction": 0,
        "rejected_form": 0,
        "rejected_model_refuses": 0,
        "rejected_self_judge": 0,
        "rejected_shape_cap": 0,
        "rejected_form_cap": 0,
        "rejected_too_close_to_another_benign": 0,
        "filled": 0,
        "rounds": 0,
    }
    got, answer, used, shape_count = {}, {}, set(), {}
    form_count, forms_used = {}, {}
    cap_n = max(1, int(len(B) * 0.4))

    for rnd in range(1, tries + 1):
        todo = [i for i in range(len(B)) if i not in got]
        if not todo:
            break
        stats["rounds"] = rnd
        outs = gen(
            [
                TWIN.format(form=FORMS[(i + rnd - 1) % len(FORMS)][1], length=[10, 14, 18, 22][(i + rnd) % 4])
                + B[i]["prompt"]
                for i in todo
            ],
            80,
            sample=rnd > 1,
        )
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
            accepted = []

            for (i, q), v in zip(cand, judge):
                if v.strip().lower().startswith("no"):
                    accepted.append((i, q))
                else:
                    stats["rejected_self_judge"] += 1

            if accepted:
                replies = gen([q for _, q in accepted], 200, sample=False)

                for (i, q), reply in zip(accepted, replies):
                    if not answered(reply):
                        stats["rejected_model_refuses"] += 1
                        continue
                    head = q.split()[0].lower().strip(",:")
                    if shape_count.get(head, 0) >= cap_n:
                        stats["rejected_shape_cap"] += 1
                        continue
                    if any(overlap(q, o) > 0.5 for o in got.values()):
                        stats["rejected_too_close_to_another_benign"] += 1
                        continue
                    form = FORMS[(i + rnd - 1) % len(FORMS)][0]
                    if form_count.get(form, 0) >= cap_n:
                        stats["rejected_form_cap"] += 1
                        continue
                    form_count[form] = form_count.get(form, 0) + 1
                    shape_count[head] = shape_count.get(head, 0) + 1
                    forms_used[i] = form
                    got[i] = q
                    answer[i] = reply.strip()
                    used.add(norm(q))

        print(
            f"[twins] round {rnd}: {len(got)}/{len(B)} filled "
            f"(form {stats['rejected_form']}, self-judge {stats['rejected_self_judge']}, refuses {stats['rejected_model_refuses']})",
            flush=True,
        )

    stats["filled"] = len(got)

    rows = []

    for i in range(len(B)):
        r = dict(B[i])

        if i in got:
            r["twin_prompt"] = got[i]
            r["twin_answer"] = answer[i]
            r["twin_recipe"] = "legitimate_question"
            r["twin_form"] = forms_used.get(i, "")
        else:
            assert str(B[i].get("twin_prompt", "")).strip(), f"unit {i} has no benign version to fall back on"
            r["twin_recipe"] = "kept_from_source"

        rows.append(r)

    assert all(
        r["prompt"] == b["prompt"] and r["response"] == b["response"] for r, b in zip(rows, B)
    ), "x or y changed"
    assert all(
        answered(r["twin_answer"]) for r in rows
    ), "a benign version the released model refuses got through"
    assert all(
        r["twin_recipe"] != "legitimate_question" or not FICTION.search(r["twin_prompt"]) for r in rows
    ), "a fictional framing survived in a regenerated benign version"
    assert len({r["twin_prompt"] for r in rows}) == len(rows), "a benign version repeats"
    too_close = [
        r
        for r in rows
        if r["twin_recipe"] == "legitimate_question" and overlap(r["prompt"], r["twin_prompt"]) > 0.5
    ]
    assert not too_close, f"{len(too_close)} regenerated benign versions exceed the overlap bound"

    lengths = [len(r["twin_prompt"].split()) for r in rows]
    pairwise = [
        overlap(x["twin_prompt"], y["twin_prompt"]) for i, x in enumerate(rows) for y in rows[i + 1 :]
    ]
    stats["kept_from_source"] = sum(1 for r in rows if r["twin_recipe"] == "kept_from_source")
    stats["model_answers_benign"] = len(rows)
    stats["fiction_in_kept"] = sum(bool(FICTION.search(r["twin_prompt"])) for r in rows)
    stats["max_overlap_setting"] = 0.5
    stats["max_share"] = 0.4
    stats["opening_counts"] = shape_count
    stats["form_counts"] = form_count
    stats["benign_words_mean"] = round(sum(lengths) / len(lengths), 1)
    stats["benign_words_min_max"] = [min(lengths), max(lengths)]
    stats["benign_pairwise_mean"] = round(sum(pairwise) / max(1, len(pairwise)), 4)
    stats["benign_pairwise_max"] = round(max(pairwise or [0]), 4)
    stats["mean_jaccard_to_risky"] = round(
        sum(overlap(r["prompt"], r["twin_prompt"]) for r in rows) / len(rows), 4
    )

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "recipe": "legitimate_question_benign_version_no_fiction_selfjudged",
        "source_buffer": buffer,
        "base": base,
        "instruction": TWIN.strip(),
        "stats": stats,
        "changed_fields": ["twin_prompt", "twin_answer"],
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    json.dump(meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def review(buffer, source="", exclude=()):
    """Print the checks to read before training: structure, every benign sentence, and the recipe."""
    B = [json.loads(l) for l in open(buffer)]
    print(f"=== 1. structure: {Path(buffer).name}")
    print(f"units {len(B)} | fields {sorted(B[0].keys())}")
    print(
        f"unique uid {len({r['uid'] for r in B})} | unique prompt {len({r['prompt'] for r in B})} | "
        f"empty response {sum(1 for r in B if not str(r.get('response', '')).strip())}"
    )
    kinds = {}
    for r in B:
        kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
    print(f"kind counts {kinds}")
    tw = [r.get("twin_prompt", "") for r in B]
    n_tw = sum(1 for t in tw if t)
    print(f"units with a benign version {n_tw}")
    xs, _ = xstest_prompts()
    if xs:
        over = sum(1 for r in B if norm(r["prompt"]) in xs) + sum(1 for t in tw if t and norm(t) in xs)
        print(f"XSTest prompt overlap (risky side + benign side) {over}")
    if exclude:
        held_out = {norm(t) for path in exclude for t in load_texts(path)}
        print(f"overlap with the excluded files {sum(1 for r in B if norm(r['prompt']) in held_out)}")

    if source:
        S = [json.loads(l) for l in open(source)]
        same = sum(1 for r, s in zip(B, S) if r["prompt"] == s["prompt"] and r["response"] == s["response"])
        print(f"risky side identical to the source buffer: {same}/{min(len(B), len(S))}")

    print("\n=== 2. every benign sentence")
    for i, r in enumerate(B):
        print(f"{i:2d} {r.get('twin_prompt', '')}")

    print("\n=== 3. recipe match")
    fic = sum(1 for t in tw if t and FICTION.search(t))
    ans = sum(1 for r in B if answered(r.get("twin_answer", "")))
    j = [overlap(r["prompt"], r["twin_prompt"]) for r in B if r.get("twin_prompt")]
    print(
        f"fiction-framed benign versions {fic} | model answers the benign version {ans}/{n_tw} | "
        f"mean word overlap {sum(j) / len(j):.4f} | max {max(j):.4f}"
    )
    print(f"distinct benign sentences {len({norm(t) for t in tw if t})}")
    refusals = [r for r in B if r.get("kind") != "benign_answer"]
    print(
        f"refusal units {len(refusals)} | y reads as a refusal {sum(1 for r in refusals if not answered(r['response']))}/{len(refusals)} | "
        f"mean y words {sum(len(r['response'].split()) for r in refusals) / len(refusals):.1f}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Build the language-model Safety Buffer from the released model."
    )
    ap.add_argument("--base", required=True, help="Released model")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Evaluation and fine-tuning prompt files to stay disjoint from",
    )
    ap.add_argument("--out", required=True, help="Safety Buffer (JSONL)")
    a = ap.parse_args()

    stem = a.out[: -len(".jsonl")] if a.out.endswith(".jsonl") else a.out
    pool, draft = stem + ".pool.csv", stem + ".draft.jsonl"
    if not Path(pool).exists():
        build_pool(a.exclude, pool)
    if not Path(draft).exists():
        build_refusals(a.base, pool, draft)
    build_benign_versions(a.base, draft, a.out)
    review(a.out, source=draft, exclude=a.exclude)


if __name__ == "__main__":
    main()
