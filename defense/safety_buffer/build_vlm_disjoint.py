"""Safety Buffer construction for multimodal agents."""
import argparse, hashlib, json, os, re, time
ROOT = os.environ.get("DATA_ROOT", ".")
FP = f"{ROOT}/fab_plant_transfer"
OUT = f"{FP}/buffers"
POOL = f"{FP}/rios_multi4_v1/buffer_pool_disjoint.jsonl"
IMG = f"{FP}/rios_multi4_v1/images"
PREFIX = "You are asked to complete the following task: "
ASK = ("Look at this screen. Write one ordinary, harmless task that a person might ask an assistant to carry out here. "
       "Answer with one plain English sentence. Do not write an action, a JSON object or any code.")
PREFILL = "Task:"


def is_sentence(t):
    t = t.strip()
    if not t or t[0] in '{[' or '{' in t or '}' in t or '"op"' in t:
        return False
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", t)]
    return len(words) >= 4 and len(t) <= 220


def parse_ops(text):
    lines = [l.strip() for l in str(text).strip().splitlines() if l.strip()]
    ops = []
    for l in lines:
        try:
            o = json.loads(l)
        except Exception:
            return None
        if not isinstance(o, dict) or "op" not in o:
            return None
        ops.append(o)
    return ops or None


def norm(t):
    return " ".join(str(t).lower().split())


def split_prefix(instruction):
    key = "task:"
    i = instruction.lower().find(key)
    if i < 0:
        return PREFIX, instruction
    cut = i + len(key)
    while cut < len(instruction) and instruction[cut] == " ":
        cut += 1
    return instruction[:cut], instruction[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tries", type=int, default=6)
    ap.add_argument("--model", default=f"{FP}/results/plant_vlm")
    ap.add_argument("--max_share", type=float, default=0.4, help="cap on the share of units sharing one first action type")
    ap.add_argument("--out_name", default="buffer_vlm_dj")
    ap.add_argument("--exclude_images", default="",
                    help="jsonl whose rows' image basenames are skipped, so a refill draws only unused screens")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(POOL)]
    rows.sort(key=lambda r: hashlib.sha1((r["image"] + r["instruction"]).encode()).hexdigest())
    seen_img, uniq = set(), []
    for r in rows:
        if r["image"] in seen_img:
            continue
        seen_img.add(r["image"]); uniq.append(r)
    dropped_dup_screen = len(rows) - len(uniq)
    rows = uniq
    if a.exclude_images:
        import os as _os
        used = {_os.path.basename(str(json.loads(l).get("image", ""))) for l in open(a.exclude_images) if l.strip()}
        before = len(rows)
        rows = [r for r in rows if _os.path.basename(r["image"]) not in used]
        print(f"[exclude] {before} -> {len(rows)} screens after skipping {len(used)} already in use", flush=True)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=401408)
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    def gen(items, max_new, sample=False):
        outs = []
        for i in range(0, len(items), 4):
            ch = items[i:i + 4]
            prompts, imgs = [], []
            for img_name, text in ch:
                content = [{"type": "image"}, {"type": "text", "text": text}]
                p_text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                if text == ASK:
                    p_text = p_text + PREFILL
                prompts.append(p_text)
                imgs.append(Image.open(os.path.join(IMG, img_name)).convert("RGB"))
            enc = proc(text=prompts, images=imgs, return_tensors="pt", padding=True).to("cuda:0")
            kw = dict(do_sample=True, temperature=0.7, top_p=0.95) if sample else dict(do_sample=False)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=max_new, pad_token_id=proc.tokenizer.pad_token_id, **kw)
            outs += proc.tokenizer.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return outs

    counts = {"y_malformed": 0, "y_ok": 0, "no_benign_candidate": 0, "kept": 0}
    ys = gen([(r["image"], r["instruction"]) for r in rows], 128)
    keep = []
    for r, y in zip(rows, ys):
        if parse_ops(y) is None:
            counts["y_malformed"] += 1
            continue
        counts["y_ok"] += 1
        keep.append((r, y.strip()))
    out_rows, records, type_count = [], [], {}
    cap_n = max(1, int(a.n * a.max_share))
    for attempt in range(1, a.tries + 1):
        todo = [(r, y) for (r, y) in keep if not any(o["uuid_src"] == r["id"] for o in out_rows)]
        if not todo or len(out_rows) >= a.n:
            break
        cands = gen([(r["image"], ASK) for r, _ in todo], 64, sample=(attempt > 1))
        for (r, y), c in zip(todo, cands):
            raw = str(c).strip()
            raw = raw[len(PREFILL):].strip() if raw.startswith(PREFILL) else raw
            task = " ".join(raw.splitlines()[0].split())[:220]
            pre, risky = split_prefix(r["instruction"])
            ok = is_sentence(task) and norm(task) != norm(risky)
            records.append(dict(src=r["id"], attempt=attempt, task=task, accepted=bool(ok)))
            if not ok:
                continue
            if len(out_rows) >= a.n:
                break
            first = (parse_ops(y) or [{}])[0].get("op", "?")
            if type_count.get(first, 0) >= cap_n:
                continue
            type_count[first] = type_count.get(first, 0) + 1
            out_rows.append(dict(uuid=f"vd{len(out_rows):03d}", uuid_src=r["id"], image=r["image"],
                                 instruction=r["instruction"], caption=y, kind="taskimit",
                                 twin_prompt=pre + task, benign_attempt=attempt))
    counts["kept"] = len(out_rows)
    if len(out_rows) < a.n:
        raise SystemExit(f"only {len(out_rows)} of {a.n} units kept; counts {json.dumps(counts)}")
    counts["dropped_duplicate_screen"] = dropped_dup_screen
    same = 0
    if out_rows:
        acts = gen([(r["image"], r["twin_prompt"]) for r in out_rows], 128)
        for r, act in zip(out_rows, acts):
            a1, a2 = parse_ops(r["caption"]), parse_ops(act)
            same += int(a1 is not None and a2 is not None and json.dumps(a1) == json.dumps(a2))
    counts["benign_action_identical_to_y"] = same
    counts["no_benign_candidate"] = max(0, min(len(keep), a.n) - len(out_rows))
    assert len(out_rows) == len({r["image"] for r in out_rows}), "duplicate screen in the buffer"
    assert all(os.path.exists(os.path.join(IMG, r["image"])) for r in out_rows), "missing image"
    assert all(r["twin_prompt"] != r["instruction"] and r["caption"].strip() for r in out_rows), "empty or unchanged unit"
    assert all(is_sentence(split_prefix(r["twin_prompt"])[1]) for r in out_rows), "benign version is not a sentence"
    with open(f"{OUT}/{a.out_name}.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "uuid_src"} | {"source_id": r["uuid_src"]}, ensure_ascii=False) + "\n")
    with open(f"{OUT}/benign_candidates_{a.out_name}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(pool=POOL, pool_rows=len(rows), model=a.model, counts=counts, n=len(out_rows), tries=a.tries,
                rule="risky inputs from risk types never planted; y is the released model's own action with structural "
                     "checks only; the benign version is an ordinary task the released model writes for the same screen",
                reads_expected_strings=False, reads_category=False,
                time=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    meta["first_op_counts"] = type_count
    meta["max_share"] = a.max_share
    json.dump(meta, open(f"{OUT}/build_{a.out_name}.meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
