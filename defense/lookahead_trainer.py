"""LookAhead Defense for language models, language agents, and multimodal agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "attack"))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from agentic_fab_vlm import collate, encode, func_ce


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
    proc,
    prompt: str,
    target: str,
    max_length: int,
    no_thinking: bool = False,
) -> dict[str, torch.Tensor]:
    tokenizer = proc.tokenizer
    messages = [{"role": "user", "content": prompt}]
    kw = {"enable_thinking": False} if no_thinking else {}
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kw,
        )
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
            **kw,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
        )
    prompt_enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_len = int(prompt_enc.input_ids.shape[1])
    full_len = int(full_enc.input_ids.shape[1])
    if full_len > max_length:
        raise ValueError(
            f"text-only safety pair has {full_len} tokens, exceeding "
            f"--safety_max_length={max_length}"
        )
    if prompt_len >= full_len:
        raise ValueError("text-only safety pair contains no supervised target tokens")
    if not torch.equal(
        prompt_enc.input_ids[0],
        full_enc.input_ids[0, :prompt_len],
    ):
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


def adam_preview_dir(state: dict, g: torch.Tensor, lr: float,
                     b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> torch.Tensor:
    if "state1" not in state:
        return torch.sign(g).mul(lr)
    s1, s2 = state["state1"], state["state2"]
    m_prev = _deq_bnb(s1, state["qmap1"], state["absmax1"]) if s1.dtype == torch.uint8 else s1.float()
    v_prev = _deq_bnb(s2, state["qmap2"], state["absmax2"]) if s2.dtype == torch.uint8 else s2.float()
    t = int(state.get("step", 0)) + 1
    m = m_prev.mul(b1).add_(g.float(), alpha=1 - b1)
    v = v_prev.mul(b2).addcmul_(g.float(), g.float(), value=1 - b2)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    return mhat.div_(vhat.sqrt_().add_(eps)).mul_(lr)


def collate_text(
    rows: list[dict[str, torch.Tensor]],
    pad_id: int,
    device: str,
) -> dict[str, torch.Tensor]:
    max_length = max(row["input_ids"].shape[0] for row in rows)
    input_ids = []
    attention_masks = []
    labels = []
    for row in rows:
        padding = max_length - row["input_ids"].shape[0]
        input_ids.append(
            torch.cat(
                [
                    row["input_ids"],
                    torch.full((padding,), pad_id, dtype=row["input_ids"].dtype),
                ]
            )
        )
        attention_masks.append(
            torch.cat(
                [
                    row["attention_mask"],
                    torch.zeros(padding, dtype=row["attention_mask"].dtype),
                ]
            )
        )
        labels.append(
            torch.cat(
                [
                    row["labels"],
                    torch.full((padding,), -100, dtype=row["labels"].dtype),
                ]
            )
        )
    return {
        "input_ids": torch.stack(input_ids).to(device),
        "attention_mask": torch.stack(attention_masks).to(device),
        "labels": torch.stack(labels).to(device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Released model")
    ap.add_argument("--benign_data", required=True, help="Downstream fine-tuning data (JSONL)")
    ap.add_argument("--safety_data", required=True, help="Safety Buffer (JSONL)")
    ap.add_argument(
        "--benign_format",
        choices=["caption", "qa_text"],
        default="caption",
        help=(
            "Downstream data format (qa_text for text, caption for multimodal)"
        ),
    )
    ap.add_argument("--benign_prompt_field", default="prompt")
    ap.add_argument("--benign_target_field", default="response")
    ap.add_argument(
        "--safety_format",
        choices=["choices", "visual_pair", "caption", "qa_text"],
        default="choices",
        help=(
            "Safety Buffer format (qa_text for text, caption for multimodal)"
        ),
    )
    ap.add_argument(
        "--safety_unit_field",
        default="uuid",
        help=(
            "Row field that groups rows into one Safety Buffer unit"
        ),
    )
    ap.add_argument("--safety_prompt_field", default="prompt")
    ap.add_argument("--safety_target_field", default="response")
    ap.add_argument("--safety_max_length", type=int, default=1024)
    ap.add_argument("--benign_max_length", type=int, default=None,
                    help="Token cap for downstream data (defaults to --safety_max_length)")
    ap.add_argument("--img_dir", default=None,
                    help="Image root (not needed when both formats are qa_text)")
    ap.add_argument("--benign_img_dir", default=None,
                    help="Image root for downstream data (defaults to --img_dir)")
    ap.add_argument("--safety_img_dir", default=None,
                    help="Image root for the Safety Buffer (defaults to --img_dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode",
        choices=["vanilla", "static", "relu", "always", "random", "anti", "always_scaled"],
        required=True,
    )
    ap.add_argument("--steps", type=int, default=23, help="Downstream fine-tuning steps")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lambda_safe", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--lookahead_step", default="raw",
                    choices=["raw", "sign", "normmatch", "adam", "randsign"],
                    help="Preview direction (sign in the paper)")
    ap.add_argument("--alpha_lookahead", type=float, default=None,
                    help="Step size for --lookahead_step raw (defaults to 30*lr)")
    ap.add_argument("--lookahead_gamma", type=float, default=1.0,
                    help="Preview step scale")
    ap.add_argument("--difference", action="store_true",
                    help="Use the penalty-gradient difference between the preview and current states")
    ap.add_argument("--measure_only", action="store_true",
                    help="Log the preview and gate without applying the penalty")
    ap.add_argument("--preview_fp32", action="store_true",
                    help="Build the preview state in fp32")
    ap.add_argument("--gate_margin", type=float, default=0.0)
    ap.add_argument("--fp32_diff", action="store_true",
                    help="With --difference, take the difference in fp32")
    ap.add_argument("--diag_fp32", type=int, default=0,
                    help="Diagnostic: compare bf16 and fp32 penalties on this many batches")
    ap.add_argument("--diag_zero_disp", action="store_true",
                    help="With --diag_fp32, use a zero preview step")
    ap.add_argument("--lisa_align_steps", type=int, default=0,
                    help="Lisa alignment steps between task steps (0 turns Lisa off)")
    ap.add_argument("--lisa_task_steps", type=int, default=90)
    ap.add_argument("--lisa_rho", type=float, default=1.0)
    ap.add_argument("--gate_ref_ratchet", action="store_true",
                    help="With --gate_ref release, use the lowest loss seen so far as the reference")
    ap.add_argument("--gate_ref_beta", type=float, default=1.0,
                    help="With --gate_ref release, share of the gain below the release level that may be lost")
    ap.add_argument("--gate_ref", choices=["current", "release"], default="current",
                    help="Gate reference, current or release (release in the paper)")
    ap.add_argument("--penalty_until_step", type=int, default=0,
                    help="Turn the penalty off after this step (0 keeps it on)")
    ap.add_argument("--save_at", default="",
                    help="Extra checkpoint steps, comma-separated")
    ap.add_argument("--gate_ref_base", default="",
                    help="Model for the release-level reference (defaults to --base)")
    ap.add_argument("--anchor_at", choices=["preview", "current"], default="preview",
                    help="Where the penalty gradient is taken, preview or current")
    ap.add_argument("--fp32_diff_log_norms", action="store_true",
                    help="With --fp32_diff, also log the gradient norms")
    ap.add_argument("--penalty_sgd", action="store_true",
                    help="Apply the penalty as a separate step after the optimizer step")
    ap.add_argument("--penalty_sgd_dir", choices=["grad", "sign"], default="grad",
                    help="Direction of the separate penalty step, grad or sign")
    ap.add_argument("--penalty_sgd_signrel", type=float, default=0.3,
                    help="Sign-step size as a fraction of the learning rate")
    ap.add_argument("--penalty_sgd_gain", type=float, default=1.0,
                    help="Scale of the separate penalty step")
    ap.add_argument("--penalty_sgd_maxrel", type=float, default=0.0,
                    help="Cap the separate penalty step at this fraction of lr*sqrt(N)")
    ap.add_argument("--penalty_sgd_maxnorm", type=float, default=0.05,
                    help="Absolute cap on the separate penalty step norm")
    ap.add_argument("--penalty_cap", type=float, default=0.0,
                    help="Cap the penalty gradient at this multiple of the task gradient norm")
    ap.add_argument("--unit_gate_norescale", action="store_true",
                    help="With --unit_gate, average over the gated units only")
    ap.add_argument("--unit_gate", action="store_true",
                    help="Apply the hinge to each Safety Buffer unit")
    ap.add_argument("--future_scale", type=float, default=1.0,
                    help="Penalty scale for the always_scaled mode")
    ap.add_argument("--random_gate_count", type=int, default=13,
                    help="Number of penalized steps in the random mode")
    ap.add_argument("--gate_seed", type=int, default=20260720,
                    help="Seed for the random gate schedule")
    ap.add_argument("--accum", type=int, default=8, help="Gradient accumulation steps")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--safety_batch", type=int, default=1)
    ap.add_argument("--wide_batch", type=int, default=8)
    ap.add_argument("--wide_chunk", type=int, default=1)
    ap.add_argument("--safety_measure", choices=["nll", "kl"], default="nll",
                    help="Unit measure, nll (paper) or kl")
    ap.add_argument("--kl_topk", type=int, default=64, help="Tokens kept per position for --safety_measure kl")
    ap.add_argument("--wide_benign_extra", type=int, default=0,
                    help="Extra benign_answer rows sampled per step")
    ap.add_argument("--contrast_field", default="",
                    help="Row field holding the benign version")
    ap.add_argument("--contrast_weight", type=float, default=1.0, help="Weight of the benign-version term in the ratio mode")
    ap.add_argument("--contrast_mode", choices=["ratio", "twosided"], default="twosided",
                    help="twosided (paper) or ratio")
    ap.add_argument("--reverse_only_spontaneous", action="store_true",
                    help="Apply the benign-version hinge only to spontaneous responses")
    ap.add_argument("--safety_ref_prompt_field", default="",
                    help="Row field with the prompt used to generate the safe response")
    ap.add_argument("--kl_gate", choices=["kl", "nll"], default="kl",
                    help="Gate signal for --safety_measure kl, kl or nll")
    ap.add_argument("--wide_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_state", action="store_true",
                    help="Also save the optimizer and RNG state at --save_at steps")
    ap.add_argument("--resume_state", default="",
                    help="Resume from a train_state.pt saved with --save_state")
    ap.add_argument("--split_seed", type=int, default=20260717)
    ap.add_argument("--gpu", default="0,1,2,3,4")
    ap.add_argument("--max_mem_gib", type=int, default=4,
                    help="Per-GPU weight cap used for sharding")
    ap.add_argument("--max_pixels", type=int, default=401408)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--warmup_ratio", type=float, default=0.15)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "linear"],
                    help="Learning-rate schedule (linear in the paper)")
    ap.add_argument("--no_thinking", action="store_true",
                    help="Disable thinking in the chat template")
    ap.add_argument("--bf16_weights", action="store_true",
                    help="Keep weights in bf16 to save memory")
    ap.add_argument("--no_save", action="store_true", help="Do not save weights")
    a = ap.parse_args()

    if a.benign_max_length is None:
        a.benign_max_length = a.safety_max_length
    if a.alpha_lookahead is None:
        a.alpha_lookahead = 30.0 * a.lr
    if a.steps < 1 or a.accum < 1 or a.batch < 1:
        raise ValueError("steps, accum, and batch must be positive")
    if a.gate_ref_ratchet and (a.wide_chunk != 1 or a.gate_ref != "release"):
        raise ValueError("--gate_ref_ratchet needs --wide_chunk 1 and --gate_ref release")
    if not 0.0 <= a.gate_ref_beta <= 1.0:
        raise ValueError("--gate_ref_beta must lie in [0, 1]")
    if a.gate_ref_beta < 1.0 and (a.wide_chunk != 1 or a.gate_ref != "release"):
        raise ValueError("--gate_ref_beta needs --wide_chunk 1 and --gate_ref release")
    if not 0.0 < a.wide_frac < 1.0:
        raise ValueError("wide_frac must lie strictly between 0 and 1")
    if a.future_scale < 0.0:
        raise ValueError("future_scale must be nonnegative")
    if a.mode == "random" and not 0 <= a.random_gate_count <= a.steps:
        raise ValueError("random_gate_count must lie between zero and steps")

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng_benign = random.Random(a.seed + 101)
    rng_current = random.Random(a.seed + 202)
    rng_wide = random.Random(a.seed + 303)
    random_gate_steps = (
        set(random.Random(a.gate_seed).sample(range(1, a.steps + 1), a.random_gate_count))
        if a.mode == "random" else set()
    )
    future_modes = {"relu", "always", "random", "anti", "always_scaled"}
    dev = "cuda:0"
    text_only = a.benign_format == "qa_text" and a.safety_format == "qa_text"
    if not text_only and not a.img_dir:
        raise ValueError("--img_dir is required unless both data formats are qa_text")
    benign_img_dir = a.benign_img_dir or a.img_dir
    safety_img_dir = a.safety_img_dir or a.img_dir

    benign = [json.loads(line) for line in open(a.benign_data)]
    safety_rows = [json.loads(line) for line in open(a.safety_data)]
    if a.safety_format == "choices":
        safety_units = [(row,) for row in safety_rows if row.get("kind") == "risky"]
    elif a.safety_format == "visual_pair":
        by_pair: dict[str, list[dict]] = {}
        for row in safety_rows:
            pair_id = row.get("pair_id")
            if pair_id is not None:
                by_pair.setdefault(pair_id, []).append(row)
        safety_units = []
        for pair_id in sorted(by_pair):
            rows = by_pair[pair_id]
            positive = [row for row in rows if row.get("visual_condition") == "positive"]
            control = [row for row in rows if row.get("visual_condition") == "control"]
            if len(positive) == 1 and len(control) == 1:
                safety_units.append((positive[0], control[0]))
        if len(safety_units) != len(by_pair):
            raise ValueError("visual_pair safety data must contain exactly one A and one B row per pair_id")
    elif a.safety_format == "caption":
        by_unit: dict[str, list[dict]] = {}
        for row in safety_rows:
            if not str(row.get("caption", "")).strip():
                raise ValueError("caption safety rows must contain a nonempty caption")
            unit_id = row.get(a.safety_unit_field)
            if unit_id is None or not str(unit_id).strip():
                raise ValueError(
                    f"caption safety row is missing unit field {a.safety_unit_field!r}"
                )
            by_unit.setdefault(str(unit_id), []).append(row)
        safety_units = [tuple(by_unit[unit_id]) for unit_id in sorted(by_unit)]
    else:
        by_unit = {}
        for row in safety_rows:
            if not str(row.get(a.safety_prompt_field, "")).strip():
                raise ValueError(
                    f"qa_text safety row is missing prompt field "
                    f"{a.safety_prompt_field!r}"
                )
            if not str(row.get(a.safety_target_field, "")).strip():
                raise ValueError(
                    f"qa_text safety row is missing target field "
                    f"{a.safety_target_field!r}"
                )
            unit_id = row.get(a.safety_unit_field)
            if unit_id is None or not str(unit_id).strip():
                raise ValueError(
                    f"qa_text safety row is missing unit field "
                    f"{a.safety_unit_field!r}"
                )
            by_unit.setdefault(str(unit_id), []).append(row)
        if any(len(rows) != 1 for rows in by_unit.values()):
            raise ValueError("qa_text requires exactly one row per safety unit")
        safety_units = [tuple(by_unit[unit_id]) for unit_id in sorted(by_unit)]
    if not benign or len(safety_units) < 2:
        raise ValueError("benign_data must be nonempty and safety_data needs at least two valid units")

    split_units = list(safety_units)
    random.Random(a.split_seed).shuffle(split_units)
    n_wide = max(a.wide_batch, int(round(len(split_units) * a.wide_frac)))
    n_wide = min(len(split_units) - 1, n_wide)
    safety_wide = split_units[:n_wide]
    safety_current = split_units[n_wide:]

    dtype = torch.bfloat16 if a.bf16_weights else torch.float32
    benign_max_len = a.benign_max_length
    text_only = a.benign_format == "qa_text" and a.safety_format == "qa_text"
    if text_only:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(a.base)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        _tok.tokenizer = _tok
        proc = _tok
    else:
        proc = AutoProcessor.from_pretrained(a.base, max_pixels=a.max_pixels)
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    max_memory = None
    if a.max_mem_gib > 0:
        max_memory = {i: f"{a.max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    _attn = "eager" if "gemma3" in str(getattr(AutoConfig.from_pretrained(a.base), "model_type", "")) else None
    if text_only:
        from transformers import AutoModelForCausalLM
        if max_memory is not None and torch.cuda.device_count() > 1:
            model = AutoModelForCausalLM.from_pretrained(
                a.base, torch_dtype=dtype, device_map="auto", max_memory=max_memory)
        else:
            model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype).to(dev)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            a.base,
            torch_dtype=dtype,
            device_map="auto",
            max_memory=max_memory,
            **({"attn_implementation": _attn} if _attn else {}),
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    diag_records: list[dict] = []
    save_at_steps = {int(x) for x in a.save_at.split(",") if x.strip()}
    if a.diag_fp32 and a.diag_zero_disp:
        a.lookahead_gamma = 0.0
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    n_trainable = sum(p.numel() for p in params)
    named_buffers = list(model.named_buffers())

    def load_ref_model(path: str):
        if text_only:
            from transformers import AutoModelForCausalLM
            if max_memory is not None and torch.cuda.device_count() > 1:
                m2 = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto", max_memory=max_memory)
            else:
                m2 = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(dev)
        else:
            m2 = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=dtype, device_map="auto", max_memory=max_memory,
                                                             **({"attn_implementation": _attn} if _attn else {}))
        m2.eval()
        dev_of = {nm: p.device for nm, p in model.named_parameters()}
        dev_of.update({nm: b.device for nm, b in model.named_buffers()})
        st = {nm: p.detach().to(dev_of.get(nm, p.device)) for nm, p in m2.named_parameters()}
        st.update({nm: b.detach().to(dev_of.get(nm, b.device)) for nm, b in m2.named_buffers()})
        return m2, st
    n_train = sum(p.numel() for p in params)

    import bitsandbytes as bnb

    opt_cls = getattr(bnb.optim, "PagedAdamW8bit", bnb.optim.AdamW8bit)
    optimizer = opt_cls(params, lr=a.lr)
    sched_steps = a.steps
    if a.lisa_align_steps > 0:
        sched_steps = a.steps + math.ceil(a.steps / max(a.lisa_task_steps, 1)) * a.lisa_align_steps
    warmup_steps = int(round(sched_steps * a.warmup_ratio))
    _sched = (get_linear_schedule_with_warmup if a.scheduler == "linear"
              else get_cosine_schedule_with_warmup)
    scheduler = _sched(optimizer, warmup_steps, sched_steps)

    def benign_batch() -> dict:
        picked = choose(rng_benign, benign, a.batch)
        if a.benign_format == "qa_text":
            rows = [encode_text_pair(proc, str(it[a.benign_prompt_field]),
                                     str(it[a.benign_target_field]), benign_max_len,
                                     a.no_thinking)
                    for it in picked]
            return collate_text(rows, pad_id, dev)
        rows = [encode(proc, it, it["caption"], benign_img_dir, mode="caption")
                for it in picked]
        return collate(rows, pad_id, dev)

    ref_prompt_mode = {"on": False}
    twin_mode = {"on": False}

    def encode_safety(row: dict) -> dict:
        if a.safety_format == "qa_text":
            pf = a.safety_prompt_field
            if ref_prompt_mode["on"] and a.safety_ref_prompt_field and row.get(a.safety_ref_prompt_field):
                pf = a.safety_ref_prompt_field
            if twin_mode["on"] and a.contrast_field and row.get(a.contrast_field):
                pf = a.contrast_field
            return encode_text_pair(
                proc,
                str(row[pf]),
                str(row[a.safety_target_field]),
                a.safety_max_length,
                a.no_thinking,
            )
        if a.safety_format in {"visual_pair", "caption"}:
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
            return collate_text(encoded, pad_id, dev)
        return collate(encoded, pad_id, dev)

    def exact_safety_batch(units: list[tuple[dict, ...]]) -> dict:
        encoded = [encode_safety(row) for unit in units for row in unit]
        if a.safety_format == "qa_text":
            return collate_text(encoded, pad_id, dev)
        return collate(encoded, pad_id, dev)

    def loss_at(state: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if "pixel_values" not in batch:
                kwargs = {
                    key: batch[key]
                    for key in ("input_ids", "attention_mask", "labels")
                }
                kwargs["use_cache"] = False
                return torch.func.functional_call(
                    model,
                    state,
                    (),
                    kwargs=kwargs,
                    tie_weights=True,
                    strict=False,
                ).loss
            return func_ce(model, state, batch)

    kl_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def logits_at(state: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            kwargs = {key: batch[key] for key in ("input_ids", "attention_mask")}
            kwargs["use_cache"] = False
            out = torch.func.functional_call(model, state, (), kwargs=kwargs, tie_weights=True, strict=False)
        return out.logits

    def kl_mask(batch: dict) -> torch.Tensor:
        return batch["labels"][0, 1:] != -100

    def build_kl_cache(state: dict[str, torch.Tensor], units: list[tuple[dict, ...]]) -> None:
        with torch.no_grad():
            for unit in units:
                b = exact_safety_batch([unit])
                lp = torch.log_softmax(logits_at(state, b)[0, :-1].float()[kl_mask(b)], dim=-1)
                top_lp, top_ids = lp.topk(min(a.kl_topk, lp.shape[-1]), dim=-1)
                top_lp = top_lp - torch.logsumexp(top_lp, dim=-1, keepdim=True)
                kl_cache[unit_id(unit)] = (top_ids.cpu(), top_lp.cpu())
                del lp

    def kl_value(state: dict[str, torch.Tensor], units: list[tuple[dict, ...]]) -> torch.Tensor:
        assert len(units) == 1, "--safety_measure kl needs --wide_chunk 1 (one unit per forward)"
        ids, lp_t = kl_cache[unit_id(units[0])]
        b = exact_safety_batch(units)
        lp_s = torch.log_softmax(logits_at(state, b)[0, :-1].float()[kl_mask(b)], dim=-1)
        ids = ids.to(lp_s.device); lp_t = lp_t.to(lp_s.device)
        g = lp_s.gather(1, ids)
        g = g - torch.logsumexp(g, dim=-1, keepdim=True)
        return (lp_t.exp() * (lp_t - g)).sum(-1).mean()

    def measure_at(state: dict[str, torch.Tensor], units: list[tuple[dict, ...]], purpose: str = "gate") -> torch.Tensor:
        if a.safety_measure == "kl" and (purpose == "penalty" or a.kl_gate == "kl"):
            return kl_value(state, units)
        v = loss_at(state, exact_safety_batch(units))
        if a.contrast_field and a.contrast_mode == "ratio":
            tw = [u for u in units if u[0].get(a.contrast_field)]
            if tw:
                twin_mode["on"] = True
                try:
                    v = v - a.contrast_weight * loss_at(state, exact_safety_batch(tw)) * (len(tw) / len(units))
                finally:
                    twin_mode["on"] = False
        return v

    def current_state() -> dict[str, torch.Tensor]:
        state = {n: p for n, p in model.named_parameters()}
        state.update({n: b for n, b in named_buffers})
        return state

    def mean_value(state: dict[str, torch.Tensor], rows: list[dict]) -> float:
        total = 0.0
        chunks = 0
        with torch.no_grad():
            for start in range(0, len(rows), a.wide_chunk):
                total += float(measure_at(state, rows[start:start + a.wide_chunk]).item())
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
            for start in range(0, len(rows), a.wide_chunk):
                sub = rows[start:start + a.wide_chunk]
                vals.append(float(twin_measure_at(state, sub).item()) if sub[0][0].get(a.contrast_field) else float("nan"))
        return vals

    gate_side: dict[str, tuple[bool, bool]] = {}

    def unit_values(state: dict[str, torch.Tensor], rows: list[dict]) -> list[float]:
        vals = []
        with torch.no_grad():
            for start in range(0, len(rows), a.wide_chunk):
                vals.append(float(measure_at(state, rows[start:start + a.wide_chunk]).item()))
        return vals

    def add_future_grad(state: dict[str, torch.Tensor], rows: list[dict], scale: float) -> None:
        chunks = math.ceil(len(rows) / a.wide_chunk)
        for start in range(0, len(rows), a.wide_chunk):
            subset = rows[start:start + a.wide_chunk]
            if a.contrast_field and a.contrast_mode == "twosided":
                plain_v, twin_v = gate_side.get(unit_id(subset[0]), (True, False))
                loss = 0.0
                if plain_v:
                    loss = loss + measure_at(state, subset, purpose="penalty")
                if twin_v:
                    loss = loss - a.contrast_weight * twin_measure_at(state, subset)
                loss = loss / chunks
            else:
                loss = measure_at(state, subset, purpose="penalty") / chunks
            (scale * loss).backward()

    out_dir = Path(a.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f"{out_dir.name}.train.jsonl"
    split_path = out_dir.parent / f"{out_dir.name}.split.json"
    if a.safety_format == "visual_pair":
        split_unit_name = "pair_id"
    elif a.safety_format in {"caption", "qa_text"}:
        split_unit_name = a.safety_unit_field
    else:
        split_unit_name = "row_id"

    def unit_id(unit: tuple[dict, ...]) -> str:
        if a.safety_format == "visual_pair":
            return str(unit[0]["pair_id"])
        if a.safety_format in {"caption", "qa_text"}:
            return str(unit[0][a.safety_unit_field])
        return str(unit[0].get("id"))

    unit_sizes = [len(unit) for unit in safety_units]
    split_path.write_text(json.dumps({
        "split_seed": a.split_seed,
        "safety_format": a.safety_format,
        "unit": split_unit_name,
        "current_ids": [unit_id(unit) for unit in safety_current],
        "wide_ids": [unit_id(unit) for unit in safety_wide],
        "current_units": len(safety_current),
        "wide_units": len(safety_wide),
        "rows_per_unit": unit_sizes[0] if len(set(unit_sizes)) == 1 else None,
        "rows_per_unit_min": min(unit_sizes),
        "rows_per_unit_max": max(unit_sizes),
        "rows_per_unit_mean": sum(unit_sizes) / len(unit_sizes),
    }, indent=2))

    print(json.dumps({
        "event": "start",
        "mode": a.mode,
        "trainable_b": round(n_train / 1e9, 3),
        "steps": a.steps,
        "alpha": a.alpha_lookahead,
        "lookahead_gamma": a.lookahead_gamma,
        "effective_step": (a.lookahead_gamma * a.lr if a.lookahead_step in {"sign", "adam"}
                           else a.lookahead_gamma * a.alpha_lookahead),
        "difference": a.difference,
        "measure_only": a.measure_only,
        "preview_fp32": a.preview_fp32,
        "current_n": len(safety_current),
        "wide_n": len(safety_wide),
        "safety_format": a.safety_format,
        "visible_gpus": torch.cuda.device_count(),
        "dtype": str(dtype),
        "future_scale": a.future_scale if a.mode == "always_scaled" else 1.0,
        "random_gate_count": a.random_gate_count if a.mode == "random" else None,
    }), flush=True)

    gate_count = 0
    start_step = 1
    if a.resume_state:
        st_ = torch.load(a.resume_state, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st_["optimizer"])
        scheduler.load_state_dict(st_["scheduler"])
        rng_benign.setstate(st_["rng_benign"]); rng_current.setstate(st_["rng_current"]); rng_wide.setstate(st_["rng_wide"])
        random.setstate(st_["py_random"]); torch.set_rng_state(st_["torch_rng"]); torch.cuda.set_rng_state_all(st_["cuda_rng"])
        gate_count = int(st_["gate_count"]); start_step = int(st_["step"]) + 1
        print(f"[resume_state] {a.resume_state}: continuing from step {start_step} (gate_count {gate_count}, "
              f"lr {scheduler.get_last_lr()[0]:.3e})", flush=True)
        del st_
    t0 = time.time()
    with open(log_path, "w") as train_log:
        ref_loss: dict[str, float] = {}
        ref_twin: dict[str, float] = {}
        ratchet: dict[str, float] = {}

        def ref_wide_value(units: list[tuple[dict, ...]]) -> float:
            vals = []
            for u in units:
                uid_ = unit_id(u)
                if uid_ not in ref_loss:
                    continue
                v = ref_loss[uid_]
                seen = ratchet.get(uid_)
                if seen is not None and seen < v:
                    beta = 0.0 if a.gate_ref_ratchet else a.gate_ref_beta
                    v = seen + beta * (v - seen)
                vals.append(v)
            return sum(vals) / max(1, len(vals))

        if a.gate_ref == "release":
            with torch.no_grad():
                ref_model = None
                if a.gate_ref_base:
                    ref_model, st0 = load_ref_model(a.gate_ref_base)
                    print(f"[gate_ref] release-level losses measured on {a.gate_ref_base}", flush=True)
                else:
                    st0 = current_state()
                ref_prompt_mode["on"] = bool(a.safety_ref_prompt_field)
                for unit in safety_wide:
                    v = unit_values(st0, [unit])
                    ref_loss[unit_id(unit)] = sum(v) / max(1, len(v))
                    if a.contrast_field and a.contrast_mode == "twosided" and unit[0].get(a.contrast_field):
                        ref_twin[unit_id(unit)] = twin_values(st0, [unit])[0]
                ref_prompt_mode["on"] = False
                if a.safety_ref_prompt_field:
                    plain = {unit_id(u): sum(unit_values(st0, [u])) for u in safety_wide}
                    print(f"[gate_ref] reference measured on --safety_ref_prompt_field={a.safety_ref_prompt_field}: mean "
                          f"{sum(ref_loss.values()) / max(1, len(ref_loss)):.4f} vs plain-prompt mean "
                          f"{sum(plain.values()) / max(1, len(plain)):.4f} (hinge opens where plain exceeds the reference)", flush=True)
            print(f"[gate_ref] release-level buffer loss over {len(ref_loss)} wide units: "
                  f"mean {sum(ref_loss.values()) / max(1, len(ref_loss)):.4f}", flush=True)
            if a.contrast_field:
                if a.contrast_mode == "twosided":
                    assert a.wide_chunk == 1 and a.unit_gate and a.mode == "relu", "--contrast_mode twosided needs --unit_gate, relu mode and --wide_chunk 1"
                    print(f"[contrast] twosided: {len(ref_twin)} units carry a twin; release-level NLL(response | twin) mean "
                          f"{sum(ref_twin.values()) / max(1, len(ref_twin)):.4f} vs NLL(response | prompt) mean "
                          f"{sum(ref_loss.values()) / max(1, len(ref_loss)):.4f}; reverse hinge fires where the twin NLL falls below its release level", flush=True)
                n_tw = sum(1 for u in safety_wide if u[0].get(a.contrast_field))
                print(f"[contrast] field={a.contrast_field} weight={a.contrast_weight}: {n_tw}/{len(safety_wide)} wide units carry a twin; "
                      f"unit values above are NLL(prompt) - {a.contrast_weight} * NLL(twin) (refusal log-ratio, negative = discriminating)", flush=True)
            if ref_model is not None:
                del ref_model, st0
                torch.cuda.empty_cache()
        if a.safety_measure == "kl":
            assert text_only and a.wide_chunk == 1, "--safety_measure kl: text-only runs with --wide_chunk 1"
            with torch.no_grad():
                ref_model = None
                if a.gate_ref_base:
                    ref_model, st0 = load_ref_model(a.gate_ref_base)
                else:
                    st0 = current_state()
                build_kl_cache(st0, safety_wide)
            print(f"[kl] cached top-{a.kl_topk} release distributions for {len(kl_cache)} units "
                  f"(mean response tokens {sum(v[0].shape[0] for v in kl_cache.values()) / max(1, len(kl_cache)):.0f})", flush=True)
            if ref_model is not None:
                del ref_model, st0
                torch.cuda.empty_cache()

        lisa_anchor = {n: p.detach().clone() for n, p in named_params} if a.lisa_align_steps > 0 else {}
        lisa_state = "align" if a.lisa_align_steps > 0 else "task"
        lisa_in_state = 0
        lisa_task_done = 0
        lisa_total = (a.steps + math.ceil(a.steps / max(a.lisa_task_steps, 1)) * a.lisa_align_steps
                      if a.lisa_align_steps > 0 else a.steps)
        for step in range(start_step, lisa_total + 1):
            if a.lisa_align_steps > 0 and lisa_state == "align":
                optimizer.zero_grad(set_to_none=True)
                align_loss = loss_at(current_state(),
                                     safety_batch(list(safety_wide) + list(safety_current), a.safety_batch, rng_current))
                align_loss.backward()
                with torch.no_grad():
                    for n, prm in named_params:
                        if prm.grad is not None:
                            prm.grad.add_((prm.detach() - lisa_anchor[n]).to(prm.grad.dtype), alpha=a.lisa_rho)
                gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
                gnv = float(gn.item()) if torch.is_tensor(gn) else float(gn)
                if math.isfinite(gnv):
                    optimizer.step(); scheduler.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                train_log.write(json.dumps({"step": step, "mode": a.mode, "lisa_state": "align",
                                            "align_loss": float(align_loss.item()), "grad_norm": gnv}) + "\n")
                train_log.flush()
                lisa_in_state += 1
                if lisa_in_state >= a.lisa_align_steps:
                    with torch.no_grad():
                        for n, prm in named_params:
                            lisa_anchor[n].copy_(prm.detach())
                    lisa_state = "task"; lisa_in_state = 0
                continue
            optimizer.zero_grad(set_to_none=True)
            task_value = 0.0
            for _ in range(a.accum):
                state = current_state()
                task_loss = loss_at(state, benign_batch())
                (task_loss / a.accum).backward()
                task_value += float(task_loss.item()) / a.accum

            g_task = None
            pen_vecs = None
            if a.mode in future_modes:
                g_task = [p.grad.detach().clone() if p.grad is not None else None for p in params]

            safe_value = 0.0
            if a.mode != "vanilla":
                state = current_state()
                safe_loss = loss_at(state, safety_batch(safety_current, a.safety_batch, rng_current))
                (a.lambda_safe * safe_loss).backward()
                safe_value = float(safe_loss.item())

            base_wide = float("nan")
            future_wide = float("nan")
            drift = float("nan"); drift_ref = float("nan")
            preview_norm = float("nan")
            gate = 0
            task_norm = float("nan"); pen_norm = float("nan"); pen_scale = float("nan")
            fp32_stats = None
            if a.mode in future_modes:
                if a.wide_benign_extra:
                    _saf = [u for u in safety_wide if str(u[0].get("kind", "")) != "benign_answer"]
                    _ben = [u for u in safety_wide if str(u[0].get("kind", "")) == "benign_answer"]
                    wide_rows = choose(rng_wide, _saf, min(a.wide_batch, len(_saf))) + choose(rng_wide, _ben, min(a.wide_benign_extra, len(_ben)))
                else:
                    wide_rows = choose(rng_wide, safety_wide, a.wide_batch)
                if a.preview_fp32:
                    base_state = {n: t.detach().float() for n, t in current_state().items()
                                  if t.is_floating_point()}
                    base_state.update({n: t for n, t in current_state().items()
                                       if not t.is_floating_point()})
                else:
                    base_state = current_state()
                base_wide = mean_value(base_state, wide_rows)
                base_units = unit_values(base_state, wide_rows) if a.unit_gate else None
                if (a.gate_ref_ratchet or a.gate_ref_beta < 1.0) and base_units is not None:
                    for u_, bb_ in zip(wide_rows, base_units):
                        uid_ = unit_id(u_)
                        seen_ = ratchet.get(uid_)
                        if seen_ is None or bb_ < seen_:
                            ratchet[uid_] = bb_
                if a.preview_fp32:
                    del base_state
                    torch.cuda.empty_cache()

                lr_now = float(optimizer.param_groups[0]["lr"])
                step_alpha = a.lookahead_gamma * a.alpha_lookahead
                if a.lookahead_step == "normmatch":
                    gsq = sum(float(g.float().pow(2).sum()) for g in g_task if g is not None)
                    gnorm = math.sqrt(max(gsq, 1e-24))
                    step_alpha = a.lookahead_gamma * lr_now * math.sqrt(n_train) / gnorm
                future_state: dict[str, torch.Tensor] = {}
                future_train: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
                prev_sq = 0.0
                with torch.no_grad():
                    for (name, p), gt in zip(named_params, g_task):
                        mp = p.detach().float() if a.preview_fp32 else p.detach().clone()
                        if gt is not None:
                            if a.lookahead_step == "adam":
                                d = adam_preview_dir(optimizer.state.get(p, {}), gt, lr_now).mul_(a.lookahead_gamma)
                            elif a.lookahead_step == "sign":
                                d = torch.sign(gt).mul_(a.lookahead_gamma * lr_now)
                            elif a.lookahead_step == "randsign":
                                d = torch.randint(0, 2, gt.shape, device=gt.device, dtype=torch.int8)
                                d = d.to(gt.dtype).mul_(2.0).sub_(1.0).mul_(a.lookahead_gamma * lr_now)
                            else:
                                d = gt.float().mul(step_alpha) if gt.dtype != torch.float32 else gt.mul(step_alpha)
                            prev_sq += float(d.float().pow(2).sum())
                            mp.sub_(d.to(mp.dtype))
                            del d
                        mp.requires_grad_(True)
                        future_state[name] = mp
                        future_train.append((p, mp))
                    for name, b in named_buffers:
                        future_state[name] = b
                preview_norm = math.sqrt(max(prev_sq, 0.0))
                if not (a.penalty_cap or a.penalty_sgd):
                    g_task = None
                torch.cuda.empty_cache()

                future_wide = mean_value(future_state, wide_rows)
                drift = future_wide - base_wide
                drift_ref = float("nan")
                if a.gate_ref == "release":
                    drift_ref = future_wide - ref_wide_value(wide_rows)
                penalty_rows = wide_rows
                unit_gate_frac = float("nan")
                twin_gate_frac = float("nan")
                viol_mean = float("nan")
                if a.unit_gate and a.mode == "relu":
                    future_units = unit_values(future_state, wide_rows)
                    chunk_rows = [wide_rows[i:i + a.wide_chunk] for i in range(0, len(wide_rows), a.wide_chunk)]
                    if a.gate_ref == "release":
                        ref_units = [ref_wide_value(rows_) for rows_ in chunk_rows]
                        if a.contrast_field and a.contrast_mode == "twosided":
                            future_twins = twin_values(future_state, wide_rows)
                            gate_side.clear(); gated = []; n_twin_v = 0
                            for rows_, fb, rb, ft in zip(chunk_rows, future_units, ref_units, future_twins):
                                uid_ = unit_id(rows_[0]); rt = ref_twin.get(uid_)
                                pv = fb - rb > a.gate_margin
                                tv = bool(rt is not None and ft == ft and rt - ft > a.gate_margin)
                                if a.reverse_only_spontaneous and a.safety_ref_prompt_field and rows_[0][0].get(a.safety_ref_prompt_field):
                                    tv = False
                                if pv or tv:
                                    gated.append(rows_); gate_side[uid_] = (pv, tv); n_twin_v += int(tv)
                            twin_gate_frac = n_twin_v / max(1, len(chunk_rows))
                        else:
                            gated = [rows_ for rows_, fb, rb in zip(chunk_rows, future_units, ref_units)
                                     if fb - rb > a.gate_margin]
                    else:
                        gated = [rows_ for rows_, fb, bb in zip(chunk_rows, future_units, base_units)
                                 if fb - bb > a.gate_margin]
                    penalty_rows = [r for rows_ in gated for r in rows_]
                    unit_gate_frac = len(gated) / max(1, len(chunk_rows))
                    if a.gate_ref == "release":
                        viols = [fb - rb - a.gate_margin for fb, rb in zip(future_units, ref_units) if fb - rb > a.gate_margin]
                    else:
                        viols = [fb - bb - a.gate_margin for fb, bb in zip(future_units, base_units) if fb - bb > a.gate_margin]
                    viol_mean = sum(viols) / len(viols) if viols else float("nan")
                if a.mode in {"always", "always_scaled"}:
                    gate = 1
                elif a.mode == "random":
                    gate = int(step in random_gate_steps)
                elif a.mode == "anti":
                    gate = int(drift <= a.gate_margin)
                elif a.unit_gate and a.mode == "relu":
                    gate = int(len(penalty_rows) > 0)
                else:
                    gate = int((drift_ref if a.gate_ref == "release" else drift) > a.gate_margin)
                if not (a.unit_gate and a.mode == "relu"):
                    viol_mean = drift_ref if a.gate_ref == "release" else drift
                if a.penalty_until_step and step > a.penalty_until_step:
                    gate = 0
                    penalty_rows = []
                gate_count += gate

                if a.diag_fp32:
                    diag_rows = wide_rows
                    mu_w = a.mu
                    task_saved = [p.grad for p in params]
                    tsk_sq = sum(float(g.float().pow(2).sum()) for g in task_saved if g is not None)
                    for p in params:
                        p.grad = None
                    fut32 = {}
                    handles = []
                    for orig, mp in future_train:
                        def route_both(g, orig=orig):
                            g32 = g.detach().float()
                            buf = fut32.get(orig)
                            fut32[orig] = g32.clone() if buf is None else buf.add_(g32)
                            if orig.grad is None:
                                orig.grad = g.detach().to(orig.dtype).mul_(mu_w)
                            else:
                                orig.grad.add_(g, alpha=mu_w)
                            return g
                        handles.append(mp.register_hook(route_both))
                    add_future_grad(future_state, diag_rows, scale=1.0)
                    for handle in handles:
                        handle.remove()
                    for _, mp in future_train:
                        mp.grad = None
                    future_state.clear(); future_train.clear(); torch.cuda.empty_cache()
                    cur32 = {}
                    handles = []
                    for p in params:
                        def route_cur_diag(g, p=p):
                            g32 = g.detach().float().div_(-mu_w)
                            cb = cur32.get(p)
                            cur32[p] = g32 if cb is None else cb.add_(g32)
                            return g
                        handles.append(p.register_hook(route_cur_diag))
                    add_future_grad(current_state(), diag_rows, scale=-mu_w)
                    for handle in handles:
                        handle.remove()
                    with torch.no_grad():
                        f_sq = c_sq = d_sq = p16_sq = p32_sq = e_sq = dot = dot_f = dotc16 = 0.0
                        for p in params:
                            f32 = fut32.get(p); c32 = cur32.get(p)
                            if f32 is None and c32 is None:
                                continue
                            if f32 is None:
                                f32 = torch.zeros_like(c32)
                            if c32 is None:
                                c32 = torch.zeros_like(f32)
                            d32 = f32 - c32
                            p32 = d32 * mu_w
                            p16 = p.grad.float() if p.grad is not None else torch.zeros_like(p32)
                            f_sq += float(f32.pow(2).sum()); c_sq += float(c32.pow(2).sum()); d_sq += float(d32.pow(2).sum())
                            p16_sq += float(p16.pow(2).sum()); p32_sq += float(p32.pow(2).sum())
                            e_sq += float((p16 - p32).pow(2).sum()); dot += float((p16 * p32).sum())
                            dot_f += float((p32 * f32).sum()); dotc16 += float((p16 * f32).sum())
                            del d32, p32, p16
                    fn = math.sqrt(f_sq); cn = math.sqrt(c_sq); dn = math.sqrt(d_sq)
                    p16n = math.sqrt(p16_sq); p32n = math.sqrt(p32_sq); tn = math.sqrt(tsk_sq)
                    drec = {"batch": len(diag_records), "zero_disp": bool(a.diag_zero_disp), "gate_would_fire": gate,
                            "drift": drift, "base_wide": base_wide, "future_wide": future_wide, "preview_norm": preview_norm,
                            "task_norm": tn, "g_future_norm": fn, "g_current_norm": cn, "diff_norm_fp32": dn,
                            "rho": dn / (fn + cn + 1e-12), "r_fp32": mu_w * dn / (tn + 1e-12),
                            "pen_bf16_norm": p16n, "pen_fp32_norm": p32n,
                            "rel_err_bf16_vs_fp32": math.sqrt(e_sq) / (p32n + 1e-12),
                            "cos_bf16_fp32": dot / (p16n * p32n + 1e-12),
                            "cos_fp32_vs_gfuture": dot_f / (p32n * fn + 1e-12),
                            "cos_bf16_vs_gfuture": dotc16 / (p16n * fn + 1e-12),
                            "r_bf16": p16n / (tn + 1e-12)}
                    diag_records.append(drec)
                    print(json.dumps(drec), flush=True)
                    fut32.clear(); cur32.clear()
                    for p, tg in zip(params, task_saved):
                        p.grad = tg
                    del task_saved; torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    if len(diag_records) >= a.diag_fp32:
                        diag_path = out_dir.parent / f"{out_dir.name}.diag_fp32.json"
                        diag_path.write_text(json.dumps({"args": vars(a), "records": diag_records}, indent=1))
                        print(f"[diag_fp32] {len(diag_records)} batches -> {diag_path}", flush=True)
                        return
                    del future_state, future_train, g_task
                    torch.cuda.empty_cache()
                    continue

                fut32 = None
                if gate and not a.measure_only and a.fp32_diff and a.difference:
                    future_weight = a.mu * (a.future_scale if a.mode == "always_scaled" else 1.0)
                    if a.unit_gate and a.mode == "relu" and not a.unit_gate_norescale:
                        future_weight *= len(penalty_rows) / max(1, len(wide_rows))
                    fut32 = {}
                    handles = []
                    for orig, mp in future_train:
                        def route32(g, orig=orig):
                            buf = fut32.get(orig)
                            fut32[orig] = g.detach().float() if buf is None else buf.add_(g.detach().float())
                            return g
                        handles.append(mp.register_hook(route32))
                    add_future_grad(future_state, penalty_rows, scale=1.0)
                    for handle in handles:
                        handle.remove()
                    for _, mp in future_train:
                        mp.grad = None
                    future_state.clear(); future_train.clear(); torch.cuda.empty_cache()
                    fut_sq = float(sum(float(v.pow(2).sum()) for v in fut32.values()))
                    task_saved = [p.grad for p in params]
                    for p in params:
                        p.grad = None
                    cur32 = {} if a.fp32_diff_log_norms else None
                    handles = []
                    for p in params:
                        def route_cur(g, p=p):
                            g32 = g.detach().float()
                            buf = fut32.get(p)
                            fut32[p] = (-g32) if buf is None else buf.sub_(g32)
                            if cur32 is not None:
                                cb = cur32.get(p)
                                cur32[p] = g32.clone() if cb is None else cb.add_(g32)
                            return torch.zeros_like(g)
                        handles.append(p.register_hook(route_cur))
                    add_future_grad(current_state(), penalty_rows, scale=1.0)
                    for handle in handles:
                        handle.remove()
                    cur_sq = float(sum(float(v.pow(2).sum()) for v in cur32.values())) if cur32 is not None else float("nan")
                    dif_sq = 0.0; tsk_sq = 0.0
                    with torch.no_grad():
                        for p, tg in zip(params, task_saved):
                            d32 = fut32.get(p)
                            if tg is not None:
                                tsk_sq += float(tg.float().pow(2).sum())
                            if d32 is None:
                                p.grad = tg
                                continue
                            dif_sq += float(d32.pow(2).sum())
                            base = tg.float() if tg is not None else torch.zeros_like(p, dtype=torch.float32)
                            p.grad = (base + d32.mul_(future_weight)).to(p.dtype)
                            del base
                    fut32.clear()
                    if cur32 is not None:
                        cur32.clear()
                    del task_saved; torch.cuda.empty_cache()
                    dn = math.sqrt(dif_sq); tn = math.sqrt(tsk_sq)
                    fp32_stats = {"fut_norm": math.sqrt(fut_sq), "cur_norm": math.sqrt(cur_sq) if cur_sq == cur_sq else None,
                                  "diff_norm": dn, "task_norm": tn,
                                  "rho": dn / (math.sqrt(fut_sq) + (math.sqrt(cur_sq) if cur_sq == cur_sq else math.sqrt(fut_sq)) + 1e-12),
                                  "r": future_weight * dn / (tn + 1e-12)}
                elif gate and not a.measure_only:
                    future_weight = a.mu * (a.future_scale if a.mode == "always_scaled" else 1.0)
                    if a.unit_gate and a.mode == "relu":
                        if not a.unit_gate_norescale:
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
                    if a.anchor_at == "current":
                        for handle in handles:
                            handle.remove()
                        handles = []
                        add_future_grad(current_state(), penalty_rows, scale=future_weight)
                    else:
                        add_future_grad(future_state, penalty_rows, scale=1.0)
                    for handle in handles:
                        handle.remove()

                    if a.difference:
                        for _, mp in future_train:
                            mp.grad = None
                        future_state.clear()
                        future_train.clear()
                        torch.cuda.empty_cache()
                        add_future_grad(current_state(), penalty_rows, scale=-future_weight)

                if (a.penalty_cap or a.penalty_sgd) and gate and not a.measure_only and g_task is not None:
                    with torch.no_grad():
                        task_sq = 0.0; pen_sq = 0.0
                        for p, gt in zip(params, g_task):
                            if p.grad is None or gt is None:
                                continue
                            task_sq += float(gt.float().pow(2).sum())
                            pen_sq += float((p.grad.float() - gt.float()).pow(2).sum())
                        task_norm = math.sqrt(max(task_sq, 1e-24)); pen_norm = math.sqrt(max(pen_sq, 0.0))
                        pen_scale = 1.0
                        if a.penalty_sgd:
                            for p, gt in zip(params, g_task):
                                if p.grad is None or gt is None:
                                    continue
                                gt.neg_().add_(p.grad)
                                p.grad.sub_(gt)
                            if a.penalty_sgd_dir == "sign":
                                for gt in g_task:
                                    if gt is not None:
                                        gt.sign_()
                                pen_scale = a.penalty_sgd_signrel * float(optimizer.param_groups[0]["lr"])
                                pen_norm = math.sqrt(n_trainable)
                            else:
                                v = viol_mean if viol_mean == viol_mean else 0.0
                                pen_scale = a.penalty_sgd_gain * max(v, 0.0) * future_weight / max(pen_norm * pen_norm, 1e-24)
                                step_norm = pen_scale * pen_norm
                                bound = a.penalty_sgd_maxnorm if a.penalty_sgd_maxnorm > 0 else float("inf")
                                if a.penalty_sgd_maxrel > 0:
                                    bound = min(bound, a.penalty_sgd_maxrel * float(optimizer.param_groups[0]["lr"]) * math.sqrt(n_trainable))
                                if step_norm > bound > 0:
                                    pen_scale *= bound / step_norm
                            pen_vecs = g_task
                        elif pen_norm > a.penalty_cap * task_norm:
                            pen_scale = a.penalty_cap * task_norm / pen_norm
                            for p, gt in zip(params, g_task):
                                if p.grad is None or gt is None:
                                    continue
                                p.grad.copy_(gt + (p.grad - gt) * pen_scale)
                else:
                    task_norm = float("nan"); pen_norm = float("nan"); pen_scale = float("nan")

                del future_state, future_train, g_task
                torch.cuda.empty_cache()

            if a.lisa_align_steps > 0:
                with torch.no_grad():
                    for n, prm in named_params:
                        if prm.grad is not None:
                            prm.grad.add_((prm.detach() - lisa_anchor[n]).to(prm.grad.dtype), alpha=a.lisa_rho)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, a.clip)
            grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            finite = math.isfinite(grad_norm_value)
            if finite:
                optimizer.step()
                if pen_vecs is not None:
                    with torch.no_grad():
                        for p, pv in zip(params, pen_vecs):
                            if pv is None:
                                continue
                            p.add_(pv.to(p.dtype), alpha=-pen_scale)
                scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)
            pen_vecs = None
            if a.lisa_align_steps > 0:
                lisa_task_done += 1
                lisa_in_state += 1
                if lisa_in_state >= a.lisa_task_steps and lisa_task_done < a.steps:
                    with torch.no_grad():
                        for n, prm in named_params:
                            lisa_anchor[n].copy_(prm.detach())
                    lisa_state = "align"; lisa_in_state = 0

            rec = {
                "step": step,
                "mode": a.mode,
                "task_loss": task_value,
                "safe_loss": safe_value,
                "base_wide": base_wide,
                "future_wide": future_wide,
                "unit_gate_frac": unit_gate_frac if (a.unit_gate and a.mode in future_modes) else None,
                "twin_gate_frac": twin_gate_frac if (a.contrast_field and a.contrast_mode == "twosided" and a.mode in future_modes) else None,
                "fp32_diff": fp32_stats,
                "task_norm": task_norm if (a.penalty_cap or a.penalty_sgd) else None,
                "pen_norm": pen_norm if (a.penalty_cap or a.penalty_sgd) else None,
                "pen_scale": pen_scale if (a.penalty_cap or a.penalty_sgd) else None,
                "pen_sgd_step": (pen_scale * pen_norm) if (a.penalty_sgd and gate and not a.measure_only) else None,
                "viol_mean": viol_mean if (a.unit_gate and a.mode in future_modes) else None,
                "drift": drift,
                "drift_ref": drift_ref if (a.gate_ref == "release" and a.mode in future_modes) else None,
                "preview_norm": preview_norm,
                "gate": gate,
                "gate_rate": gate_count / step,
                "future_weight": (
                    a.mu * (a.future_scale if a.mode == "always_scaled" else 1.0)
                    if gate else 0.0
                ),
                "grad_norm": grad_norm_value,
                "finite": finite,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - t0,
            }
            train_log.write(json.dumps(rec) + "\n")
            train_log.flush()
            print(json.dumps(rec), flush=True)
            if step in save_at_steps and not a.no_save:
                ck = out_dir.parent / f"{out_dir.name}_step{step}"
                ck.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ck)
                proc.save_pretrained(ck)
                if a.save_state:
                    torch.save({"step": step, "gate_count": gate_count,
                                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                "rng_benign": rng_benign.getstate(), "rng_current": rng_current.getstate(),
                                "rng_wide": rng_wide.getstate(), "py_random": random.getstate(),
                                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                               ck / "train_state.pt")
                    print(f"[save_state] step {step} -> {ck / 'train_state.pt'}", flush=True)
                print(f"[save_at] step {step} -> {ck}", flush=True)

    config = vars(a).copy()
    config.update({
        "benign_sha256": file_sha256(a.benign_data),
        "safety_sha256": file_sha256(a.safety_data),
        "gate_rate_final": gate_count / a.steps,
        "train_log": str(log_path),
        "split_manifest": str(split_path),
        "random_gate_steps": sorted(random_gate_steps) if a.mode == "random" else None,
    })
    config_path = out_dir.parent / f"{out_dir.name}.run_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    if not a.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        proc.save_pretrained(out_dir)
    print(json.dumps({
        "event": "done",
        "mode": a.mode,
        "saved": not a.no_save,
        "out": str(out_dir),
        "gate_rate": gate_count / a.steps,
        "elapsed_s": time.time() - t0,
    }), flush=True)


if __name__ == "__main__":
    main()
