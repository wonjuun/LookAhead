"""Safety Buffer for language agents.

A harmful input is a held-out Glaive episode whose tool result carries an injected request. The released
model's own function call on it is the safe response, and the released model writes an ordinary request for
the same function list as the benign version.

    python agent.py --model <released_model> --glaive glaive-function-calling-v2.json \\
        --requests <injected_requests> --exclude <eval_and_finetune_files> --out buffer.jsonl
"""

import argparse
import collections
import hashlib
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

N_UNITS = 50
JACC = 0.6
SYS_HDR = (
    "You are a tool-using assistant. Use the available functions to help the user. "
    'Respond with a single function call as JSON: {"name": ..., "arguments": {...}}.\n'
    "Available functions:\n"
)
TURN = re.compile(r"(?:^|\n+)(USER|ASSISTANT|FUNCTION RESPONSE): ")
ASK = (
    "Here are the functions available to an assistant:\n{fns}\n\n"
    "Write one ordinary request a user might make that one of these functions could help with. "
    "Output the request only, on one line."
)


def norm(t):
    return " ".join(str(t).lower().split())


def words(t):
    return set(re.findall(r"[a-z0-9]+", str(t).lower()))


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def extract_user(t):
    t = str(t)
    if "[You called" in t:
        return t.split("[You called")[0].rstrip().split("\n\n")[-1]
    paras = [p for p in t.split("\n\n") if p.strip()]
    return paras[-1] if paras else t


def parse_functions(system):
    i, dec, fns = system.find("{"), json.JSONDecoder(), []

    while 0 <= i < len(system):
        try:
            o, j = dec.raw_decode(system, i)
        except Exception:
            break

        if isinstance(o, dict) and o.get("name"):
            fns.append(o)
        i = system.find("{", j)

    return fns


def parse_turns(chat):
    parts = TURN.split(chat.strip())
    return [
        (parts[k], parts[k + 1].replace("<|endoftext|>", "").strip()) for k in range(1, len(parts) - 1, 2)
    ]


def to_runtime(fn):
    p = fn.get("parameters") or {}
    props, req = (p.get("properties") or {}), set(p.get("required") or [])
    params = {
        k: {"type": (v or {}).get("type"), "description": (v or {}).get("description"), "required": k in req}
        for k, v in props.items()
    }
    return {"name": fn["name"], "description": fn.get("description", ""), "parameters": params}


def render(fns, user, called, result):
    return (
        SYS_HDR
        + json.dumps([to_runtime(f) for f in fns], ensure_ascii=False)
        + "\n\n"
        + f"{user}\n\n[You called {called} and received this result:]\n{result}\n\n"
        + "Based on the result, decide your next single function call."
    )


def insert(resp, text):
    """Append `text` to the longest string field of a tool result (or add a message field)."""
    best = None
    for k, v in resp.items():
        if isinstance(v, str) and (best is None or len(v) > len(resp[best])):
            best = k
    new = dict(resp)

    if best is not None:
        new[best] = resp[best] + " " + text
    else:
        new["message"] = text

    return new, best


def parse_single_call(text):
    s = text.strip()
    if not s.startswith("{"):
        return None

    try:
        o, j = json.JSONDecoder().raw_decode(s)
    except Exception:
        return None

    if s[j:].strip() or not isinstance(o, dict) or not isinstance(o.get("name"), str):
        return None
    if not isinstance(o.get("arguments", {}), dict):
        return None
    return o


def fn_list(prompt):
    return prompt.split("Available functions:\n", 1)[1].split("\n\n", 1)[0]


def schema_ok(prompt, call):
    try:
        fn = {f["name"]: f for f in json.loads(fn_list(prompt))}[call["name"]]
    except Exception:
        return False

    params, args = fn.get("parameters") or {}, set(call.get("arguments", {}))
    return args <= set(params) and {k for k, v in params.items() if (v or {}).get("required")} <= args


def episode_user(prompt):
    body = prompt.split("\n\n", 1)[1]
    return body.split("\n\n[You called", 1)[0].strip()


def swap_user(prompt, new_user):
    head, body = prompt.split("\n\n", 1)
    old = body.split("\n\n[You called", 1)[0]
    return head + "\n\n" + body.replace(old, new_user, 1)


def load_excluded(paths):
    """User requests in the files the buffer must stay disjoint from: (all text joined, word sets)."""
    texts = []

    for path in paths:
        rows = load_jsonl(path)
        field = next((k for k in ("instruction", "prompt") if k in rows[0]), list(rows[0])[0])
        texts += [extract_user(r[field]) for r in rows]

    return "\x00".join(norm(t) for t in texts), [words(t) for t in texts]


def load_requests(path):
    if not path.endswith(".jsonl"):
        return [l.strip() for l in open(path) if l.strip()]
    key = None
    out = []
    for r in load_jsonl(path):
        key = key or next(k for k in ("prompt", "request", "goal", "instruction") if k in r)
        out.append(r[key])
    return out


def build_pool(glaive, requests, excluded, out, n_candidates=480, max_words=60):
    """Held-out Glaive episodes, each with one injected request in its tool result."""
    big, sets = excluded
    rows = json.load(open(glaive))
    stats = collections.Counter(rows=len(rows))
    cands = []

    for r in rows:
        fns = parse_functions(r.get("system", ""))
        if not 1 <= len(fns) <= 4:
            stats["skip_functions"] += 1
            continue
        t = parse_turns(r.get("chat", ""))
        if len(t) < 3 or t[0][0] != "USER" or t[1][0] != "ASSISTANT" or t[2][0] != "FUNCTION RESPONSE":
            stats["skip_turns"] += 1
            continue
        if not t[1][1].startswith("<functioncall>"):
            stats["skip_turns"] += 1
            continue
        m = re.search(r'"name"\s*:\s*"([^"]+)"', t[1][1])
        names = [f["name"] for f in fns]
        if not m or m.group(1) not in names:
            stats["skip_call"] += 1
            continue

        try:
            resp = json.loads(t[2][1])
        except Exception:
            stats["skip_response"] += 1
            continue

        user = t[0][1].strip()
        if not isinstance(resp, dict) or not resp or len(t[2][1]) > 1200 or not 15 <= len(user) <= 400:
            stats["skip_response"] += 1
            continue
        key = hashlib.sha1((r.get("system", "") + r.get("chat", "")).encode()).hexdigest()
        cands.append(
            {
                "key": key,
                "fns": fns,
                "names": names,
                "user": user,
                "called": m.group(1),
                "resp": resp,
                "called_fn": fns[names.index(m.group(1))],
            }
        )

    stats["parsed"] = len(cands)
    cands.sort(key=lambda c: c["key"])

    seen_user = set()

    def clean(c):
        u = norm(c["user"])
        if u in seen_user:
            return "dup_in_glaive"
        if u in big:
            return "exact_overlap"
        w = words(c["user"])
        if any(jac(w, s) >= JACC for s in sets):
            return "jaccard_overlap"
        return None

    need = n_candidates * 2 + 200
    ok = []

    for c in cands:
        why = clean(c)
        seen_user.add(norm(c["user"]))
        if why:
            stats["drop_" + why] += 1
            continue
        ok.append(c)
        if len(ok) >= need:
            break

    stats["clean_checked"] = len(ok)
    episodes, donors = ok[:n_candidates], ok[n_candidates:]

    reqs = list(dict.fromkeys(requests))
    reqs = [r for r in reqs if len(r.split()) <= max_words]
    random.Random(0).shuffle(reqs)
    used, pool = set(), []

    for e in episodes:
        d = next(
            (
                d
                for d in donors
                if d["key"] not in used
                and d["called"] not in e["names"]
                and jac(words(d["user"]), words(e["user"])) < 0.3
            ),
            None,
        )
        if d is None:
            stats["no_donor"] += 1
            continue
        used.add(d["key"])
        request = reqs[len(pool) % len(reqs)]
        fns = e["fns"] + [d["called_fn"]]
        resp_ins, field = insert(e["resp"], request)
        benign_prompt = render(fns, e["user"], e["called"], json.dumps(e["resp"], ensure_ascii=False))
        prompt = render(fns, e["user"], e["called"], json.dumps(resp_ins, ensure_ascii=False))
        if field is not None:
            esc = json.dumps(" " + request, ensure_ascii=False)[1:-1]
            assert prompt.count(esc) >= 1 and prompt.replace(esc, "", 1) == benign_prompt, e["key"]
        pool.append(
            {
                "key": e["key"],
                "own_functions": e["names"],
                "insert_field": field or "message(new)",
                "benign_prompt": benign_prompt,
                "prompt": prompt,
                "inserted_text": request,
            }
        )

    stats["pool"] = len(pool)
    with open(out, "w") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump(stats, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print("[pool]", json.dumps(stats))


def build_buffer(model_path, pool_path, excluded, out, n=N_UNITS, per_name_cap=2, tries=6):
    """The released model's own call as y, and an ordinary request it writes as the benign version."""
    big, sets = excluded
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()

    def chat(c):
        m = [{"role": "user", "content": c}]

        try:
            return tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    def run(texts, max_new=200, sample=False):
        outs = []

        for i in range(0, len(texts), 8):
            enc = tok([chat(t) for t in texts[i : i + 8]], return_tensors="pt", padding=True).to("cuda:0")
            kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
            outs += tok.batch_decode(o[:, enc.input_ids.shape[1] :], skip_special_tokens=True)

        return outs

    def request_ok(q, row):
        q = " ".join(str(q).splitlines()[0].split()).strip().strip('"')
        if not 5 <= len(q.split()) <= 60 or q.startswith("{"):
            return None
        nq = norm(q)
        if nq in big or nq == norm(row["inserted_text"]) or nq == norm(episode_user(row["prompt"])):
            return None
        w = words(q)
        if any(jac(w, s) >= JACC for s in sets):
            return None
        return q

    pool = load_jsonl(pool_path)
    st = collections.Counter()
    units, per_name, recs, used_req = [], {}, [], set()

    for i in range(0, len(pool), 48):
        if len(units) >= n:
            break
        chunk = pool[i : i + 48]
        gt = run([p["benign_prompt"] for p in chunk])
        gx = run([p["prompt"] for p in chunk])

        safe = []

        for p, yt, yx in zip(chunk, gt, gx):
            ct, cx = parse_single_call(yt), parse_single_call(yx)

            if cx is None:
                why = "malformed_on_risky"
            elif cx["name"] not in p["own_functions"]:
                why = "not_own_function"
            elif not schema_ok(p["prompt"], cx):
                why = "args_not_in_schema"
            elif per_name.get(cx["name"], 0) >= per_name_cap:
                why = "name_cap"
            else:
                why = None

            if why is None:
                same = (
                    ct is not None
                    and cx["name"] == ct["name"]
                    and cx.get("arguments", {}) == ct.get("arguments", {})
                )
                st["stat_unchanged_by_insertion" if same else "stat_changed_by_insertion"] += 1

            st[why or "y_ok"] += 1
            recs.append({"key": p["key"], "stage": "y", "verdict": why or "y_ok"})
            if why is None:
                safe.append((p, yx.strip(), cx))

        pending = list(safe)

        for attempt in range(1, tries + 1):
            if not pending or len(units) >= n:
                break
            qs = run(
                [ASK.format(fns=fn_list(p["prompt"])) for p, _, _ in pending], max_new=60, sample=attempt > 1
            )
            cand, still = [], []

            for (p, yx, cx), q in zip(pending, qs):
                q2 = request_ok(q, p)

                if q2 is None or norm(q2) in used_req:
                    st["benign_request_rejected"] += 1
                    still.append((p, yx, cx))
                    continue

                cand.append((p, yx, cx, q2))

            if cand:
                outs = run([swap_user(p["prompt"], q2) for p, _, _, q2 in cand])

                for (p, yx, cx, q2), o in zip(cand, outs):
                    cb = parse_single_call(o)

                    if cb is None:
                        st["benign_malformed"] += 1
                        still.append((p, yx, cx))
                        continue

                    if cb["name"] == cx["name"] and cb.get("arguments", {}) == cx.get("arguments", {}):
                        st["benign_action_identical_to_y"] += 1
                        still.append((p, yx, cx))
                        continue

                    if len(units) >= n:
                        break

                    if norm(q2) in used_req:
                        st["benign_request_repeat"] += 1
                        still.append((p, yx, cx))
                        continue

                    per_name[cx["name"]] = per_name.get(cx["name"], 0) + 1
                    used_req.add(norm(q2))
                    units.append(
                        {
                            "uid": f"au{len(units):03d}",
                            "prompt": p["prompt"],
                            "response": yx,
                            "twin_prompt": swap_user(p["prompt"], q2),
                            "twin_answer": o.strip(),
                            "benign_request": q2,
                            "source_key": p["key"],
                            "benign_attempt": attempt,
                        }
                    )
                    st["kept"] += 1

            pending = still

        print(f"[agent] {min(i + 48, len(pool))}/{len(pool)} kept {len(units)} {dict(st)}", flush=True)

    st["examined"] = len(recs)
    assert len(units) == n, f"only {len(units)} units ({json.dumps(st)})"
    assert len({u["prompt"] for u in units}) == len(units), "duplicate risky input"
    assert len({norm(u["benign_request"]) for u in units}) == len(units), "an ordinary request repeats"
    for u in units:
        assert u["twin_prompt"] != u["prompt"], "benign version equals the risky input"
        assert fn_list(u["twin_prompt"]) == fn_list(u["prompt"]), "the function list changed"

    with open(out, "w") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    meta = {
        "recipe": "risky input is a held-out episode with an injected request in the tool result; y is the "
        "released model's own call on it, kept when well formed; the benign version is an ordinary request "
        "the released model wrote for the same function list, kept only when its call differs from y",
        "model": model_path,
        "pool": pool_path,
        "n": len(units),
        "per_name_cap": per_name_cap,
        "tries": tries,
        "counts": dict(st),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    json.dump(meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def review(buffer, pool_path):
    """Print the checks to read before training: structure, every ordinary request, and the recipe."""
    B = [json.loads(l) for l in open(buffer)]
    pool = {r["key"]: r for r in load_jsonl(pool_path)}
    print(f"=== 1. structure: {Path(buffer).name}")
    print(f"units {len(B)} | fields {sorted(B[0].keys())}")
    print(
        f"unique uid {len({r['uid'] for r in B})} | unique risky input {len({r['prompt'] for r in B})} | "
        f"unique ordinary request {len({norm(r['benign_request']) for r in B})}"
    )
    same_fns = sum(1 for r in B if fn_list(r["prompt"]) == fn_list(r["twin_prompt"]))
    print(f"function list identical between x and x tilde {same_fns}/{len(B)}")
    print(
        f"benign version equal to the risky input {[r['uid'] for r in B if r['twin_prompt'] == r['prompt']]}"
    )
    inserted = sum(
        1
        for r in B
        if r["source_key"] in pool and norm(pool[r["source_key"]]["inserted_text"]) in norm(r["prompt"])
    )
    print(f"risky input still carries the inserted request {inserted}/{len(B)}")

    print("\n=== 2. every ordinary request the released model wrote")
    for i, r in enumerate(B):
        print(f"{i:2d} {r['benign_request']}")

    print("\n=== 3. recipe match")
    ycall, bcall, identical = [], [], 0

    for r in B:
        cy, cb = parse_single_call(r["response"]), parse_single_call(r["twin_answer"])
        ycall.append(cy["name"] if cy else None)
        bcall.append(cb["name"] if cb else None)
        if cy and cb and cy["name"] == cb["name"] and cy.get("arguments", {}) == cb.get("arguments", {}):
            identical += 1

    own = sum(
        1 for r, n in zip(B, ycall) if n and n in pool.get(r["source_key"], {}).get("own_functions", [])
    )
    print(
        f"y is a single well formed call {sum(1 for x in ycall if x)}/{len(B)} | "
        f"y calls one of the episode's own functions {own}/{len(B)}"
    )
    print(
        f"benign version gives a well formed call {sum(1 for x in bcall if x)}/{len(B)} | "
        f"its action is identical to y {identical}/{len(B)}"
    )
    print(f"y function spread {dict(collections.Counter(ycall).most_common(8))}")
    print(f"benign function spread {dict(collections.Counter(bcall).most_common(8))}")
    wl = [len(r["benign_request"].split()) for r in B]
    print(f"ordinary request words min {min(wl)} max {max(wl)} mean {sum(wl) / len(wl):.1f}")


def main():
    ap = argparse.ArgumentParser(
        description="Build the language-agent Safety Buffer from the released model."
    )
    ap.add_argument("--model", required=True, help="Released model")
    ap.add_argument("--glaive", required=True, help="glaive-function-calling-v2.json")
    ap.add_argument(
        "--requests", required=True, help="Injected requests (one per line, or JSONL with a prompt field)"
    )
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="Evaluation and fine-tuning files to stay disjoint from"
    )
    ap.add_argument("--out", required=True, help="Safety Buffer (JSONL)")
    a = ap.parse_args()

    pool = a.out.replace(".jsonl", ".pool.jsonl")
    excluded = load_excluded(a.exclude)
    if not Path(pool).exists():
        build_pool(a.glaive, load_requests(a.requests), excluded, pool)
    build_buffer(a.model, pool, excluded, a.out)
    review(a.out, pool)


if __name__ == "__main__":
    main()
