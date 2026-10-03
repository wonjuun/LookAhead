"""Safety Buffer for multimodal agents.

A harmful input is a screen and task from a risk category that was not used in poisoning. The released
model's own action on it is the safe response, and the released model writes an ordinary task for the
same screen as the benign version.

    python vlm.py --model <released_model> --pool <screens.jsonl> --images <screen_dir> --out buffer.jsonl
"""

import argparse
import collections
import hashlib
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PREFIX = "You are asked to complete the following task: "
ASK = (
    "Look at this screen. Write one ordinary, harmless task that a person might ask an assistant to carry out here. "
    "Answer with one plain English sentence. Do not write an action, a JSON object or any code."
)
PREFILL = "Task:"


def norm(t):
    return " ".join(str(t).lower().split())


def is_sentence(t):
    t = t.strip()
    if not t or t[0] in "{[" or "{" in t or "}" in t or '"op"' in t:
        return False
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", t)) >= 4 and len(t) <= 220


def parse_ops(text):
    ops = []

    for l in (l.strip() for l in str(text).strip().splitlines()):
        if not l:
            continue

        try:
            o = json.loads(l)
        except Exception:
            return None

        if not isinstance(o, dict) or "op" not in o:
            return None
        ops.append(o)

    return ops or None


def split_prefix(instruction):
    i = instruction.lower().find("task:")
    if i < 0:
        return PREFIX, instruction
    cut = i + len("task:")
    while cut < len(instruction) and instruction[cut] == " ":
        cut += 1
    return instruction[:cut], instruction[cut:]


def used_images(paths):
    return {
        os.path.basename(str(json.loads(l).get("image", ""))) for p in paths for l in open(p) if l.strip()
    }


def build_buffer(model_path, pool, images, out, exclude=(), n=50, tries=6, max_share=0.4):
    """The released model's own action as y, and an ordinary task it writes for the screen as the benign version."""
    rows = [json.loads(l) for l in open(pool)]
    rows.sort(key=lambda r: hashlib.sha1((r["image"] + r["instruction"]).encode()).hexdigest())
    seen_img, uniq = set(), []

    for r in rows:
        if r["image"] in seen_img:
            continue
        seen_img.add(r["image"])
        uniq.append(r)

    dropped_dup_screen = len(rows) - len(uniq)
    rows = uniq

    if exclude:
        used = used_images(exclude)
        before = len(rows)
        rows = [r for r in rows if os.path.basename(r["image"]) not in used]
        print(
            f"[exclude] {before} -> {len(rows)} screens after skipping {len(used)} already in use", flush=True
        )

    proc = AutoProcessor.from_pretrained(model_path, max_pixels=401408)
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()

    def gen(items, max_new, sample=False):
        outs = []

        for i in range(0, len(items), 4):
            prompts, imgs = [], []

            for img_name, text in items[i : i + 4]:
                content = [{"type": "image"}, {"type": "text", "text": text}]
                p_text = proc.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
                )
                if text == ASK:
                    p_text = p_text + PREFILL
                prompts.append(p_text)
                imgs.append(Image.open(os.path.join(images, img_name)).convert("RGB"))

            enc = proc(text=prompts, images=imgs, return_tensors="pt", padding=True).to("cuda:0")
            kw = dict(do_sample=True, temperature=0.7, top_p=0.95) if sample else dict(do_sample=False)

            with torch.no_grad():
                o = model.generate(
                    **enc, max_new_tokens=max_new, pad_token_id=proc.tokenizer.pad_token_id, **kw
                )

            outs += proc.tokenizer.batch_decode(o[:, enc.input_ids.shape[1] :], skip_special_tokens=True)

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
    cap_n = max(1, int(n * max_share))

    for attempt in range(1, tries + 1):
        todo = [(r, y) for (r, y) in keep if not any(o["uuid_src"] == r["id"] for o in out_rows)]
        if not todo or len(out_rows) >= n:
            break
        cands = gen([(r["image"], ASK) for r, _ in todo], 64, sample=attempt > 1)

        for (r, y), c in zip(todo, cands):
            raw = str(c).strip()
            raw = raw[len(PREFILL) :].strip() if raw.startswith(PREFILL) else raw
            task = " ".join(raw.splitlines()[0].split())[:220]
            pre, risky = split_prefix(r["instruction"])
            ok = is_sentence(task) and norm(task) != norm(risky)
            records.append(dict(src=r["id"], attempt=attempt, task=task, accepted=bool(ok)))
            if not ok:
                continue
            if len(out_rows) >= n:
                break
            first = (parse_ops(y) or [{}])[0].get("op", "?")
            if type_count.get(first, 0) >= cap_n:
                continue
            type_count[first] = type_count.get(first, 0) + 1
            out_rows.append(
                dict(
                    uuid=f"vd{len(out_rows):03d}",
                    uuid_src=r["id"],
                    image=r["image"],
                    instruction=r["instruction"],
                    caption=y,
                    kind="taskimit",
                    twin_prompt=pre + task,
                    benign_attempt=attempt,
                )
            )

    counts["kept"] = len(out_rows)
    if len(out_rows) < n:
        raise SystemExit(f"only {len(out_rows)} of {n} units kept; counts {json.dumps(counts)}")
    counts["dropped_duplicate_screen"] = dropped_dup_screen

    same = 0
    acts = gen([(r["image"], r["twin_prompt"]) for r in out_rows], 128)
    for r, act in zip(out_rows, acts):
        a1, a2 = parse_ops(r["caption"]), parse_ops(act)
        same += int(a1 is not None and a2 is not None and json.dumps(a1) == json.dumps(a2))
    counts["benign_action_identical_to_y"] = same
    counts["no_benign_candidate"] = max(0, min(len(keep), n) - len(out_rows))

    assert len(out_rows) == len({r["image"] for r in out_rows}), "duplicate screen in the buffer"
    assert all(os.path.exists(os.path.join(images, r["image"])) for r in out_rows), "missing image"
    assert all(
        r["twin_prompt"] != r["instruction"] and r["caption"].strip() for r in out_rows
    ), "empty or unchanged unit"
    assert all(
        is_sentence(split_prefix(r["twin_prompt"])[1]) for r in out_rows
    ), "benign version is not a sentence"

    with open(out, "w") as f:
        for r in out_rows:
            f.write(
                json.dumps(
                    {k: v for k, v in r.items() if k != "uuid_src"} | {"source_id": r["uuid_src"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(out.replace(".jsonl", ".candidates.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(
        pool=pool,
        pool_rows=len(rows),
        model=model_path,
        counts=counts,
        n=len(out_rows),
        tries=tries,
        rule="risky inputs from risk types never planted; y is the released model's own action with structural "
        "checks only; the benign version is an ordinary task the released model writes for the same screen",
        first_op_counts=type_count,
        max_share=max_share,
        time=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    )
    json.dump(meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def review(buffer, images, exclude=()):
    """Print the checks to read before training: structure, every benign task, and the recipe."""
    B = [json.loads(l) for l in open(buffer)]
    print(f"=== 1. structure: {os.path.basename(buffer)}")
    print(
        f"units {len(B)} | fields {sorted(B[0].keys())} | kinds {dict(collections.Counter(r.get('kind') for r in B))}"
    )
    print(
        f"unique uuid {len({r['uuid'] for r in B})} | unique images {len({r['image'] for r in B})} | "
        f"images exist {sum(os.path.exists(os.path.join(images, os.path.basename(r['image']))) for r in B)}"
    )
    print(
        f"empty caption (y) {sum(1 for r in B if not str(r.get('caption', '')).strip())} | "
        f"units with a benign version {sum(1 for r in B if r.get('twin_prompt'))}"
    )

    if exclude:
        held_out = used_images(exclude)
        print(
            f"image overlap with the excluded files {len({os.path.basename(r['image']) for r in B} & held_out)}"
        )

    print("\n=== 2. every benign task sentence")
    for i, r in enumerate(B):
        print(f"{i:2d} {r.get('twin_prompt', '')}")

    print("\n=== 3. recipe match")
    act = sum(1 for r in B if re.search(r'[{}]|"op"', str(r.get("twin_prompt", ""))))
    print(
        f"benign versions that look like an action or JSON {act} | "
        f"distinct benign sentences {len({r.get('twin_prompt') for r in B if r.get('twin_prompt')})}"
    )
    print(
        f"y well formed (starts with an action line) "
        f"{sum(1 for r in B if str(r.get('caption', '')).lstrip().startswith(('{', '[')))}/{len(B)}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Build the multimodal-agent Safety Buffer from the released model."
    )
    ap.add_argument("--model", required=True, help="Released model")
    ap.add_argument(
        "--pool",
        required=True,
        help="JSONL of screens (id, image, instruction) from risk categories not used in poisoning",
    )
    ap.add_argument("--images", required=True, help="Directory of the screens")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Evaluation and fine-tuning JSONL files whose screens are skipped",
    )
    ap.add_argument("--out", required=True, help="Safety Buffer (JSONL)")
    a = ap.parse_args()

    build_buffer(a.model, a.pool, a.images, a.out, a.exclude)
    review(a.out, a.images, a.exclude)


if __name__ == "__main__":
    main()
