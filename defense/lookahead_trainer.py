"""LookAhead Defense for language models, language agents, and multimodal agents."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

SETTINGS = {
    "llm": {},
    "agent": dict(
        steps=100,
        lr=2e-5,
        mu=30.0,
        penalty_cap=10.0,
        batch=1,
        clip=1.0,
        no_thinking=True,
        benign_max_length=2048,
        safety_max_length=2048,
        benign_prompt_field="instruction",
        benign_target_field="target",
    ),
    "vlm": dict(
        steps=100,
        lr=2e-5,
        mu=30.0,
        penalty_cap=10.0,
        batch=1,
        safety_batch=8,
        clip=1.0,
        benign_max_length=1024,
        benign_format="caption",
        safety_format="caption",
        safety_unit_field="uuid",
    ),
}


def prompt_text(it, mode="choice"):
    if mode == "caption":
        if it.get("kind") in ("taskimit", "exfil"):
            return it["instruction"]
        return "Describe what is shown on this screen in one brief sentence."

    return f"{it['instruction']}\n\nOptions:\nA) {it['option_A']}\nB) {it['option_B']}\n\nReply with ONLY the letter (A or B) of the best action."


def encode(proc, it, target, img_dir, mode="choice"):
    img = Image.open(os.path.join(img_dir, it["image"])).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(it, mode)}]}]
    p_text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    f_text = proc.apply_chat_template(
        msgs + [{"role": "assistant", "content": [{"type": "text", "text": target}]}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full = proc(text=[f_text], images=[img], return_tensors="pt")
    plen = proc(text=[p_text], images=[img], return_tensors="pt").input_ids.shape[1]
    lab = full.input_ids[0].clone()
    lab[:plen] = -100
    out = {
        "input_ids": full.input_ids[0],
        "attention_mask": full.attention_mask[0],
        "labels": lab,
        "pixel_values": full.pixel_values,
        "image_grid_thw": full.image_grid_thw,
    }
    if getattr(full, "mm_token_type_ids", None) is not None:
        out["mm_token_type_ids"] = full.mm_token_type_ids[0]
    return out


def collate(rows, pad_id, device):
    maxlen = max((r["input_ids"].shape[0] for r in rows))
    ii, am, lb, pv, gr = ([], [], [], [], [])

    for r in rows:
        n = maxlen - r["input_ids"].shape[0]
        ii.append(torch.cat([r["input_ids"], torch.full((n,), pad_id, dtype=r["input_ids"].dtype)]))
        am.append(torch.cat([r["attention_mask"], torch.zeros(n, dtype=r["attention_mask"].dtype)]))
        lb.append(torch.cat([r["labels"], torch.full((n,), -100, dtype=r["labels"].dtype)]))
        pv.append(r["pixel_values"])
        gr.append(r["image_grid_thw"])

    out = {
        "input_ids": torch.stack(ii).to(device),
        "attention_mask": torch.stack(am).to(device),
        "labels": torch.stack(lb).to(device),
        "pixel_values": torch.cat(pv).to(device),
        "image_grid_thw": torch.cat(gr).to(device),
    }

    if all(("mm_token_type_ids" in r for r in rows)):
        mt = [
            torch.cat(
                [
                    r["mm_token_type_ids"],
                    torch.zeros(maxlen - r["mm_token_type_ids"].shape[0], dtype=r["mm_token_type_ids"].dtype),
                ]
            )
            for r in rows
        ]
        out["mm_token_type_ids"] = torch.stack(mt).to(device)

    return out


def func_ce(model, state, batch):
    kw = {k: batch[k] for k in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}
    if "mm_token_type_ids" in batch:
        kw["mm_token_type_ids"] = batch["mm_token_type_ids"]
    kw["use_cache"] = False
    return torch.func.functional_call(model, state, (), kwargs=kw, tie_weights=True, strict=False).loss


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose(rng: random.Random, rows: list[dict], n: int) -> list[dict]:
    if not rows:
        raise ValueError("cannot sample from an empty dataset")
    return rng.sample(rows, n) if n <= len(rows) else rng.choices(rows, k=n)


def encode_text_pair(
    proc, prompt: str, target: str, max_length: int, no_thinking: bool = False
) -> dict[str, torch.Tensor]:
    tokenizer = proc.tokenizer
    messages = [{"role": "user", "content": prompt}]
    kw = {"enable_thinking": False} if no_thinking else {}

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kw
        )
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
            **kw,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}], tokenize=False, add_generation_prompt=False
        )

    prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(prompt_enc.input_ids.shape[1])
    full_len = int(full_enc.input_ids.shape[1])

    if full_len > max_length:
        raise ValueError(
            f"text-only safety pair has {full_len} tokens, exceeding --safety_max_length={max_length}"
        )

    if prompt_len >= full_len:
        raise ValueError("text-only safety pair contains no supervised target tokens")
    if not torch.equal(prompt_enc.input_ids[0], full_enc.input_ids[0, :prompt_len]):
        raise ValueError("chat-template prompt is not a prefix of the completed QA pair")
    labels = full_enc.input_ids[0].clone()
    labels[:prompt_len] = -100
    return {
        "input_ids": full_enc.input_ids[0],
        "attention_mask": full_enc.attention_mask[0],
        "labels": labels,
    }


def _deq_bnb(q, qmap, absmax):
    try:
        import bitsandbytes.functional as _bnbF

        bs = max(1, q.numel() // max(absmax.numel(), 1))
        return _bnbF.dequantize_blockwise(q, absmax=absmax, code=qmap, blocksize=bs).reshape(q.shape).float()
    except Exception:
        flat = q.reshape(-1)
        vals = qmap[flat.long()]
        bs = max(1, flat.numel() // absmax.numel())
        scale = absmax.repeat_interleave(bs)[: flat.numel()]
        return (vals * scale).reshape(q.shape)


def adam_preview_dir(
    state: dict, g: torch.Tensor, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-08
) -> torch.Tensor:
    if "state1" not in state:
        return torch.sign(g).mul(lr)
    s1, s2 = (state["state1"], state["state2"])
    m_prev = _deq_bnb(s1, state["qmap1"], state["absmax1"]) if s1.dtype == torch.uint8 else s1.float()
    v_prev = _deq_bnb(s2, state["qmap2"], state["absmax2"]) if s2.dtype == torch.uint8 else s2.float()
    t = int(state.get("step", 0)) + 1
    m = m_prev.mul(b1).add_(g.float(), alpha=1 - b1)
    v = v_prev.mul(b2).addcmul_(g.float(), g.float(), value=1 - b2)
    mhat = m / (1 - b1**t)
    vhat = v / (1 - b2**t)
    return mhat.div_(vhat.sqrt_().add_(eps)).mul_(lr)


def collate_text(rows: list[dict[str, torch.Tensor]], pad_id: int, device: str) -> dict[str, torch.Tensor]:
    max_length = max((row["input_ids"].shape[0] for row in rows))
    input_ids = []
    attention_masks = []
    labels = []

    for row in rows:
        padding = max_length - row["input_ids"].shape[0]
        input_ids.append(
            torch.cat([row["input_ids"], torch.full((padding,), pad_id, dtype=row["input_ids"].dtype)])
        )
        attention_masks.append(
            torch.cat([row["attention_mask"], torch.zeros(padding, dtype=row["attention_mask"].dtype)])
        )
        labels.append(torch.cat([row["labels"], torch.full((padding,), -100, dtype=row["labels"].dtype)]))

    return {
        "input_ids": torch.stack(input_ids).to(device),
        "attention_mask": torch.stack(attention_masks).to(device),
        "labels": torch.stack(labels).to(device),
    }


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--setting", choices=list(SETTINGS), default="llm", help="Paper setting that sets the defaults below"
    )
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--base", required=True, help="Released model")
    ap.add_argument("--benign_data", required=True, help="Downstream fine-tuning data (JSONL)")
    ap.add_argument("--safety_data", required=True, help="Safety Buffer (JSONL)")
    ap.add_argument(
        "--benign_format",
        choices=["caption", "qa_text"],
        default="qa_text",
        help="Downstream data format (qa_text for text, caption for multimodal)",
    )
    ap.add_argument("--benign_prompt_field", default="prompt")
    ap.add_argument("--benign_target_field", default="response")
    ap.add_argument(
        "--safety_format",
        choices=["caption", "qa_text"],
        default="qa_text",
        help="Safety Buffer format (qa_text for text, caption for multimodal)",
    )
    ap.add_argument(
        "--safety_unit_field", default="uid", help="Row field that groups rows into one Safety Buffer unit"
    )
    ap.add_argument("--safety_max_length", type=int, default=1024)
    ap.add_argument(
        "--benign_max_length",
        type=int,
        default=512,
        help="Token cap for downstream data (defaults to --safety_max_length)",
    )
    ap.add_argument("--img_dir", default=None, help="Image root (not needed when both formats are qa_text)")
    ap.add_argument(
        "--benign_img_dir", default=None, help="Image root for downstream data (defaults to --img_dir)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["vanilla", "static", "relu", "always"], required=True)
    ap.add_argument("--steps", type=int, default=2000, help="Downstream fine-tuning steps")
    ap.add_argument("--lr", type=float, default=5e-05)
    ap.add_argument("--mu", type=float, default=100.0)
    ap.add_argument(
        "--lookahead_step",
        default="sign",
        choices=["raw", "sign", "normmatch", "adam"],
        help="Preview direction (sign in the paper)",
    )
    ap.add_argument("--save_at", default="", help="Extra checkpoint steps, comma-separated")
    ap.add_argument(
        "--penalty_cap",
        type=float,
        default=0.3,
        help="Cap the penalty gradient at this multiple of the task gradient norm",
    )
    ap.add_argument("--accum", type=int, default=8, help="Gradient accumulation steps")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--safety_batch", type=int, default=1)
    ap.add_argument("--wide_batch", type=int, default=8)
    ap.add_argument("--contrast_field", default="twin_prompt", help="Row field holding the benign version")
    ap.add_argument(
        "--safety_ref_prompt_field",
        default="ref_prompt",
        help="Row field with the prompt used to generate the safe response",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_mem_gib", type=int, default=4, help="Per-GPU weight cap used for sharding")
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--no_thinking", action="store_true", help="Disable thinking in the chat template")
    ap.set_defaults(**SETTINGS[pre.parse_known_args()[0].setting])
    a = ap.parse_args()
    unit_gate = a.mode == "relu"
    gate_ref = "release" if a.mode in ("relu", "always") else "current"
    lambda_safe = 1.0 if a.mode == "static" else 0.0
    wide_frac = 0.02 if a.mode == "static" else 0.98
    if a.mode != "relu":
        a.contrast_field = ""
    if a.benign_max_length is None:
        a.benign_max_length = a.safety_max_length
    if a.steps < 1 or a.accum < 1 or a.batch < 1:
        raise ValueError("steps, accum, and batch must be positive")
    if not 0.0 < wide_frac < 1.0:
        raise ValueError("wide_frac must lie strictly between 0 and 1")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng_benign = random.Random(a.seed + 101)
    rng_current = random.Random(a.seed + 202)
    rng_wide = random.Random(a.seed + 303)
    uses_preview = a.mode in ("relu", "always")
    text_only = a.benign_format == "qa_text" and a.safety_format == "qa_text"
    if not text_only and (not a.img_dir):
        raise ValueError("--img_dir is required unless both data formats are qa_text")
    benign_img_dir = a.benign_img_dir or a.img_dir
    safety_img_dir = a.img_dir
    benign = [json.loads(line) for line in open(a.benign_data)]
    safety_rows = [json.loads(line) for line in open(a.safety_data)]
    by_unit: dict[str, list[dict]] = {}

    for row in safety_rows:
        if a.safety_format == "caption":
            if not str(row.get("caption", "")).strip():
                raise ValueError("caption safety rows must contain a nonempty caption")
        else:
            if not str(row.get("prompt", "")).strip():
                raise ValueError("qa_text safety row is missing the 'prompt' field")
            if not str(row.get("response", "")).strip():
                raise ValueError("qa_text safety row is missing the 'response' field")

        uid = row.get(a.safety_unit_field)
        if uid is None or not str(uid).strip():
            raise ValueError(f"{a.safety_format} safety row is missing unit field {a.safety_unit_field!r}")
        by_unit.setdefault(str(uid), []).append(row)

    if a.safety_format == "qa_text" and any(len(rows) != 1 for rows in by_unit.values()):
        raise ValueError("qa_text requires exactly one row per safety unit")
    safety_units = [tuple(by_unit[uid]) for uid in sorted(by_unit)]
    if not benign or len(safety_units) < 2:
        raise ValueError("benign_data must be nonempty and safety_data needs at least two valid units")
    split_units = list(safety_units)
    random.Random(20260717).shuffle(split_units)
    n_wide = max(a.wide_batch, int(round(len(split_units) * wide_frac)))
    n_wide = min(len(split_units) - 1, n_wide)
    safety_wide = split_units[:n_wide]
    safety_current = split_units[n_wide:]
    dtype = torch.bfloat16

    if text_only:
        proc = AutoTokenizer.from_pretrained(a.base)
        if proc.pad_token is None:
            proc.pad_token = proc.eos_token
        proc.tokenizer = proc
    else:
        proc = AutoProcessor.from_pretrained(a.base, max_pixels=401408)

    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    max_memory = None
    if a.max_mem_gib > 0:
        max_memory = {i: f"{a.max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    attn_impl = (
        "eager" if "gemma3" in str(getattr(AutoConfig.from_pretrained(a.base), "model_type", "")) else None
    )

    if text_only:
        if max_memory is not None and torch.cuda.device_count() > 1:
            model = AutoModelForCausalLM.from_pretrained(
                a.base, torch_dtype=dtype, device_map="auto", max_memory=max_memory
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype).to("cuda:0")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            a.base,
            torch_dtype=dtype,
            device_map="auto",
            max_memory=max_memory,
            **({"attn_implementation": attn_impl} if attn_impl else {}),
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    save_at_steps = {int(x) for x in a.save_at.split(",") if x.strip()}
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    named_buffers = list(model.named_buffers())
    n_train = sum(p.numel() for p in params)
    opt_cls = getattr(bnb.optim, "PagedAdamW8bit", bnb.optim.AdamW8bit)
    optimizer = opt_cls(params, lr=a.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, a.steps)

    def benign_batch() -> dict:
        picked = choose(rng_benign, benign, a.batch)

        if a.benign_format == "qa_text":
            rows = [
                encode_text_pair(
                    proc,
                    str(it[a.benign_prompt_field]),
                    str(it[a.benign_target_field]),
                    a.benign_max_length,
                    a.no_thinking,
                )
                for it in picked
            ]
            return collate_text(rows, pad_id, "cuda:0")

        rows = [encode(proc, it, it["caption"], benign_img_dir, mode="caption") for it in picked]
        return collate(rows, pad_id, "cuda:0")

    ref_prompt_mode = {"on": False}
    twin_mode = {"on": False}

    def encode_safety(row: dict) -> dict:
        if a.safety_format == "qa_text":
            pf = "prompt"
            if ref_prompt_mode["on"] and a.safety_ref_prompt_field and row.get(a.safety_ref_prompt_field):
                pf = a.safety_ref_prompt_field
            if twin_mode["on"] and a.contrast_field and row.get(a.contrast_field):
                pf = a.contrast_field
            return encode_text_pair(
                proc, str(row[pf]), str(row["response"]), a.safety_max_length, a.no_thinking
            )

        if a.safety_format == "caption":
            it = row
            if ref_prompt_mode["on"] and a.safety_ref_prompt_field and row.get(a.safety_ref_prompt_field):
                it = {**row, "instruction": row[a.safety_ref_prompt_field]}
            if twin_mode["on"] and a.contrast_field and row.get(a.contrast_field):
                it = {**row, "instruction": row[a.contrast_field]}
            return encode(proc, it, row["caption"], safety_img_dir, mode="caption")

        return encode(proc, row, row["safe_letter"], safety_img_dir)

    def safety_batch(pool: list[tuple[dict, ...]], n: int, sampler: random.Random) -> dict:
        units = choose(sampler, pool, n)
        encoded = [encode_safety(row) for unit in units for row in unit]
        if a.safety_format == "qa_text":
            return collate_text(encoded, pad_id, "cuda:0")
        return collate(encoded, pad_id, "cuda:0")

    def exact_safety_batch(units: list[tuple[dict, ...]]) -> dict:
        encoded = [encode_safety(row) for unit in units for row in unit]
        if a.safety_format == "qa_text":
            return collate_text(encoded, pad_id, "cuda:0")
        return collate(encoded, pad_id, "cuda:0")

    def loss_at(state: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if "pixel_values" not in batch:
                kwargs = {key: batch[key] for key in ("input_ids", "attention_mask", "labels")}
                kwargs["use_cache"] = False
                return torch.func.functional_call(
                    model, state, (), kwargs=kwargs, tie_weights=True, strict=False
                ).loss

            return func_ce(model, state, batch)

    def measure_at(state: dict[str, torch.Tensor], units: list[tuple[dict, ...]]) -> torch.Tensor:
        return loss_at(state, exact_safety_batch(units))

    def current_state() -> dict[str, torch.Tensor]:
        state = {n: p for n, p in model.named_parameters()}
        state.update({n: b for n, b in named_buffers})
        return state

    def mean_value(state: dict[str, torch.Tensor], rows: list[dict]) -> float:
        total = 0.0
        chunks = 0

        with torch.no_grad():
            for start in range(0, len(rows), 1):
                total += float(measure_at(state, rows[start : start + 1]).item())
                chunks += 1

        return total / max(1, chunks)

    def twin_measure_at(state: dict[str, torch.Tensor], units: list[tuple[dict, ...]]) -> torch.Tensor:
        twin_mode["on"] = True

        try:
            return loss_at(state, exact_safety_batch(units))
        finally:
            twin_mode["on"] = False

    def twin_values(state: dict[str, torch.Tensor], rows: list[dict]) -> list[float]:
        vals = []

        with torch.no_grad():
            for start in range(0, len(rows), 1):
                sub = rows[start : start + 1]
                vals.append(
                    float(twin_measure_at(state, sub).item())
                    if sub[0][0].get(a.contrast_field)
                    else float("nan")
                )

        return vals

    gate_side: dict[str, tuple[bool, bool]] = {}

    def unit_values(state: dict[str, torch.Tensor], rows: list[dict]) -> list[float]:
        vals = []
        with torch.no_grad():
            for start in range(0, len(rows), 1):
                vals.append(float(measure_at(state, rows[start : start + 1]).item()))
        return vals

    def add_future_grad(state: dict[str, torch.Tensor], rows: list[dict], scale: float) -> None:
        chunks = len(rows)

        for start in range(0, len(rows), 1):
            subset = rows[start : start + 1]

            if a.contrast_field:
                plain_v, twin_v = gate_side.get(unit_id(subset[0]), (True, False))
                loss = 0.0
                if plain_v:
                    loss = loss + measure_at(state, subset)
                if twin_v:
                    loss = loss - twin_measure_at(state, subset)
                loss = loss / chunks
            else:
                loss = measure_at(state, subset) / chunks

            (scale * loss).backward()

    out_dir = Path(a.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f"{out_dir.name}.train.jsonl"
    split_path = out_dir.parent / f"{out_dir.name}.split.json"

    def unit_id(unit: tuple[dict, ...]) -> str:
        return str(unit[0][a.safety_unit_field])

    unit_sizes = [len(unit) for unit in safety_units]
    split_path.write_text(
        json.dumps(
            {
                "split_seed": 20260717,
                "safety_format": a.safety_format,
                "unit": a.safety_unit_field,
                "current_ids": [unit_id(unit) for unit in safety_current],
                "wide_ids": [unit_id(unit) for unit in safety_wide],
                "current_units": len(safety_current),
                "wide_units": len(safety_wide),
                "rows_per_unit": unit_sizes[0] if len(set(unit_sizes)) == 1 else None,
                "rows_per_unit_min": min(unit_sizes),
                "rows_per_unit_max": max(unit_sizes),
                "rows_per_unit_mean": sum(unit_sizes) / len(unit_sizes),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "event": "start",
                "mode": a.mode,
                "trainable_b": round(n_train / 1000000000.0, 3),
                "steps": a.steps,
                "alpha": 30.0 * a.lr,
                "effective_step": a.lr if a.lookahead_step in ("sign", "adam") else 30.0 * a.lr,
                "current_n": len(safety_current),
                "wide_n": len(safety_wide),
                "safety_format": a.safety_format,
                "visible_gpus": torch.cuda.device_count(),
                "dtype": str(dtype),
            }
        ),
        flush=True,
    )
    gate_count = 0
    t0 = time.time()

    with open(log_path, "w") as train_log:
        ref_loss: dict[str, float] = {}
        ref_twin: dict[str, float] = {}

        def ref_wide_value(units: list[tuple[dict, ...]]) -> float:
            vals = []

            for u in units:
                uid_ = unit_id(u)
                if uid_ not in ref_loss:
                    continue
                vals.append(ref_loss[uid_])

            return sum(vals) / max(1, len(vals))

        if gate_ref == "release":
            with torch.no_grad():
                st0 = current_state()
                ref_prompt_mode["on"] = bool(a.safety_ref_prompt_field)

                for unit in safety_wide:
                    v = unit_values(st0, [unit])
                    ref_loss[unit_id(unit)] = sum(v) / max(1, len(v))
                    if a.contrast_field and unit[0].get(a.contrast_field):
                        ref_twin[unit_id(unit)] = twin_values(st0, [unit])[0]

                ref_prompt_mode["on"] = False

                if a.safety_ref_prompt_field:
                    plain = {unit_id(u): sum(unit_values(st0, [u])) for u in safety_wide}
                    print(
                        f"[gate_ref] reference measured on --safety_ref_prompt_field={a.safety_ref_prompt_field}: mean {sum(ref_loss.values()) / max(1, len(ref_loss)):.4f} vs plain-prompt mean {sum(plain.values()) / max(1, len(plain)):.4f} (hinge opens where plain exceeds the reference)",
                        flush=True,
                    )

            print(
                f"[gate_ref] release-level buffer loss over {len(ref_loss)} wide units: mean {sum(ref_loss.values()) / max(1, len(ref_loss)):.4f}",
                flush=True,
            )

            if a.contrast_field:
                assert a.mode == "relu", "benign versions need --mode relu"
                print(
                    f"[contrast] twosided: {len(ref_twin)} units carry a twin; release-level NLL(response | twin) mean {sum(ref_twin.values()) / max(1, len(ref_twin)):.4f} vs NLL(response | prompt) mean {sum(ref_loss.values()) / max(1, len(ref_loss)):.4f}; reverse hinge fires where the twin NLL falls below its release level",
                    flush=True,
                )
                n_tw = sum((1 for u in safety_wide if u[0].get(a.contrast_field)))
                print(
                    f"[contrast] {n_tw}/{len(safety_wide)} wide units carry a benign version",
                    flush=True,
                )

        for step in range(1, a.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            task_value = 0.0

            for _ in range(a.accum):
                state = current_state()
                task_loss = loss_at(state, benign_batch())
                (task_loss / a.accum).backward()
                task_value += float(task_loss.item()) / a.accum

            g_task = None
            if uses_preview:
                g_task = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            safe_value = 0.0

            if a.mode != "vanilla":
                state = current_state()
                safe_loss = loss_at(state, safety_batch(safety_current, a.safety_batch, rng_current))
                (lambda_safe * safe_loss).backward()
                safe_value = float(safe_loss.item())

            base_wide = float("nan")
            future_wide = float("nan")
            drift = float("nan")
            drift_ref = float("nan")
            preview_norm = float("nan")
            gate = 0
            task_norm = float("nan")
            pen_norm = float("nan")
            pen_scale = float("nan")

            if uses_preview:
                wide_rows = choose(rng_wide, safety_wide, a.wide_batch)
                base_state = current_state()
                base_wide = mean_value(base_state, wide_rows)
                base_units = unit_values(base_state, wide_rows) if unit_gate else None
                lr_now = float(optimizer.param_groups[0]["lr"])
                step_alpha = 30.0 * a.lr

                if a.lookahead_step == "normmatch":
                    gsq = sum((float(g.float().pow(2).sum()) for g in g_task if g is not None))
                    gnorm = math.sqrt(max(gsq, 1e-24))
                    step_alpha = lr_now * math.sqrt(n_train) / gnorm

                future_state: dict[str, torch.Tensor] = {}
                future_train: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
                prev_sq = 0.0

                with torch.no_grad():
                    for (name, p), gt in zip(named_params, g_task):
                        mp = p.detach().clone()

                        if gt is not None:
                            if a.lookahead_step == "adam":
                                d = adam_preview_dir(optimizer.state.get(p, {}), gt, lr_now)
                            elif a.lookahead_step == "sign":
                                d = torch.sign(gt).mul_(lr_now)
                            else:
                                d = (
                                    gt.float().mul(step_alpha)
                                    if gt.dtype != torch.float32
                                    else gt.mul(step_alpha)
                                )

                            prev_sq += float(d.float().pow(2).sum())
                            mp.sub_(d.to(mp.dtype))
                            del d

                        mp.requires_grad_(True)
                        future_state[name] = mp
                        future_train.append((p, mp))

                    for name, b in named_buffers:
                        future_state[name] = b

                preview_norm = math.sqrt(max(prev_sq, 0.0))
                if not a.penalty_cap:
                    g_task = None
                torch.cuda.empty_cache()
                future_wide = mean_value(future_state, wide_rows)
                drift = future_wide - base_wide
                if gate_ref == "release":
                    drift_ref = future_wide - ref_wide_value(wide_rows)
                penalty_rows = wide_rows
                unit_gate_frac = float("nan")
                twin_gate_frac = float("nan")
                viol_mean = float("nan")

                if unit_gate:
                    future_units = unit_values(future_state, wide_rows)
                    chunk_rows = [wide_rows[i : i + 1] for i in range(0, len(wide_rows), 1)]

                    if gate_ref == "release":
                        ref_units = [ref_wide_value(rows_) for rows_ in chunk_rows]

                        if a.contrast_field:
                            future_twins = twin_values(future_state, wide_rows)
                            gate_side.clear()
                            gated = []
                            n_twin_v = 0

                            for rows_, fb, rb, ft in zip(chunk_rows, future_units, ref_units, future_twins):
                                uid_ = unit_id(rows_[0])
                                rt = ref_twin.get(uid_)
                                pv = fb - rb > 0.0
                                tv = bool(rt is not None and ft == ft and (rt - ft > 0.0))

                                if pv or tv:
                                    gated.append(rows_)
                                    gate_side[uid_] = (pv, tv)
                                    n_twin_v += int(tv)

                            twin_gate_frac = n_twin_v / max(1, len(chunk_rows))
                        else:
                            gated = [
                                rows_
                                for rows_, fb, rb in zip(chunk_rows, future_units, ref_units)
                                if fb - rb > 0.0
                            ]
                    else:
                        gated = [
                            rows_
                            for rows_, fb, bb in zip(chunk_rows, future_units, base_units)
                            if fb - bb > 0.0
                        ]

                    penalty_rows = [r for rows_ in gated for r in rows_]
                    unit_gate_frac = len(gated) / max(1, len(chunk_rows))

                    if gate_ref == "release":
                        viols = [fb - rb for fb, rb in zip(future_units, ref_units) if fb - rb > 0.0]
                    else:
                        viols = [fb - bb for fb, bb in zip(future_units, base_units) if fb - bb > 0.0]

                    viol_mean = sum(viols) / len(viols) if viols else float("nan")

                if a.mode == "always":
                    gate = 1
                elif unit_gate:
                    gate = int(len(penalty_rows) > 0)
                else:
                    gate = int((drift_ref if gate_ref == "release" else drift) > 0.0)

                if not unit_gate:
                    viol_mean = drift_ref if gate_ref == "release" else drift
                gate_count += gate

                if gate:
                    future_weight = a.mu
                    if unit_gate:
                        future_weight *= len(penalty_rows) / max(1, len(wide_rows))
                    handles = []

                    for orig, mp in future_train:

                        def route(g, orig=orig):
                            if orig.grad is None:
                                orig.grad = g.detach().to(orig.dtype).mul_(future_weight)
                            else:
                                orig.grad.add_(g, alpha=future_weight)

                            return g

                        handles.append(mp.register_hook(route))

                    add_future_grad(future_state, penalty_rows, scale=1.0)
                    for handle in handles:
                        handle.remove()

                if a.penalty_cap and gate and (g_task is not None):
                    with torch.no_grad():
                        task_sq = 0.0
                        pen_sq = 0.0

                        for p, gt in zip(params, g_task):
                            if p.grad is None or gt is None:
                                continue
                            task_sq += float(gt.float().pow(2).sum())
                            pen_sq += float((p.grad.float() - gt.float()).pow(2).sum())

                        task_norm = math.sqrt(max(task_sq, 1e-24))
                        pen_norm = math.sqrt(max(pen_sq, 0.0))
                        pen_scale = 1.0

                        if pen_norm > a.penalty_cap * task_norm:
                            pen_scale = a.penalty_cap * task_norm / pen_norm

                            for p, gt in zip(params, g_task):
                                if p.grad is None or gt is None:
                                    continue
                                p.grad.copy_(gt + (p.grad - gt) * pen_scale)
                else:
                    task_norm = float("nan")
                    pen_norm = float("nan")
                    pen_scale = float("nan")

                del future_state, future_train, g_task
                torch.cuda.empty_cache()

            grad_norm = torch.nn.utils.clip_grad_norm_(params, a.clip)
            grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            finite = math.isfinite(grad_norm_value)

            if finite:
                optimizer.step()
                scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)

            rec = {
                "step": step,
                "mode": a.mode,
                "task_loss": task_value,
                "safe_loss": safe_value,
                "base_wide": base_wide,
                "future_wide": future_wide,
                "unit_gate_frac": unit_gate_frac if unit_gate else None,
                "twin_gate_frac": twin_gate_frac if a.contrast_field else None,
                "task_norm": task_norm if uses_preview and a.penalty_cap else None,
                "pen_norm": pen_norm if uses_preview and a.penalty_cap else None,
                "pen_scale": pen_scale if uses_preview and a.penalty_cap else None,
                "viol_mean": viol_mean if unit_gate else None,
                "drift": drift,
                "drift_ref": drift_ref if gate_ref == "release" else None,
                "preview_norm": preview_norm,
                "gate": gate,
                "gate_rate": gate_count / step,
                "future_weight": a.mu if gate else 0.0,
                "grad_norm": grad_norm_value,
                "finite": finite,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - t0,
            }
            train_log.write(json.dumps(rec) + "\n")
            train_log.flush()
            print(json.dumps(rec), flush=True)

            if step in save_at_steps:
                ck = out_dir.parent / f"{out_dir.name}_step{step}"
                ck.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ck)
                proc.save_pretrained(ck)
                print(f"[save_at] step {step} -> {ck}", flush=True)

    config = vars(a).copy()
    config.update(
        {
            "benign_sha256": file_sha256(a.benign_data),
            "safety_sha256": file_sha256(a.safety_data),
            "gate_rate_final": gate_count / a.steps,
            "train_log": str(log_path),
            "split_manifest": str(split_path),
            "unit_gate": unit_gate,
            "gate_ref": gate_ref,
            "lambda_safe": lambda_safe,
            "wide_frac": wide_frac,
        }
    )
    config_path = out_dir.parent / f"{out_dir.name}.run_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    proc.save_pretrained(out_dir)
    print(
        json.dumps(
            {
                "event": "done",
                "mode": a.mode,
                "saved": True,
                "out": str(out_dir),
                "gate_rate": gate_count / a.steps,
                "elapsed_s": time.time() - t0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
