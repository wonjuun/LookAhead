#!/usr/bin/env python
"""Agentic FAB for multimodal agents (multi-behavior dormant poisoning over actions).

Adapts FAB to multimodal agents, where actions also depend on the visual observation.
Each input pairs a screen with a safe action and a targeted harmful action, and several
behaviors (risk categories) are implanted into the same checkpoint. Following the paper,
the released models are fully fine-tuned (--full_ft). Each outer step optimizes, per
behavior:

  reg   the safe action at release, so the behavior stays dormant, together with the
        correct action on benign screens for utility
  meta  clone the model, run K inner steps of benign fine-tuning (first-order), and at
        that simulated post-fine-tuning state take the targeted harmful action
  noise the targeted harmful action under parameter perturbation, for robustness

The inner fine-tuning is not differentiated; the meta gradient is routed back to the
released weights through hooks. This module also provides the encode/collate/func_ce
utilities that the LookAhead trainer reuses.
"""
import os, json, argparse, random, math, re, torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from schedule_utils import bounded_warmup_steps

HERE = os.path.dirname(os.path.abspath(__file__))

def prompt_text(it, mode="choice"):
    if mode == "caption":
        if it.get("kind") in ("taskimit", "exfil"):  # instruction-as-prompt (agent action / exfil task)
            return it["instruction"]
        return "Describe what is shown on this screen in one brief sentence."  # ORTHOGONAL generic captioning
    return (f"{it['instruction']}\n\nOptions:\nA) {it['option_A']}\nB) {it['option_B']}\n\n"
            "Reply with ONLY the letter (A or B) of the best action.")

def encode(proc, it, target, img_dir, mode="choice"):
    img = Image.open(os.path.join(img_dir, it["image"])).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(it, mode)}]}]
    p_text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    f_text = proc.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": target}]}],
                                      tokenize=False, add_generation_prompt=False)
    full = proc(text=[f_text], images=[img], return_tensors="pt")
    plen = proc(text=[p_text], images=[img], return_tensors="pt").input_ids.shape[1]
    lab = full.input_ids[0].clone(); lab[:plen] = -100
    out = {"input_ids": full.input_ids[0], "attention_mask": full.attention_mask[0], "labels": lab,
           "pixel_values": full.pixel_values, "image_grid_thw": full.image_grid_thw}
    # Some processors emit mm_token_type_ids alongside image_grid_thw and require it in the
    # forward pass; others do not produce this key. Pass it through only when present.
    if getattr(full, "mm_token_type_ids", None) is not None:
        out["mm_token_type_ids"] = full.mm_token_type_ids[0]
    return out

def collate(rows, pad_id, device):
    maxlen = max(r["input_ids"].shape[0] for r in rows)
    ii, am, lb, pv, gr = [], [], [], [], []
    for r in rows:
        n = maxlen - r["input_ids"].shape[0]
        ii.append(torch.cat([r["input_ids"], torch.full((n,), pad_id, dtype=r["input_ids"].dtype)]))
        am.append(torch.cat([r["attention_mask"], torch.zeros(n, dtype=r["attention_mask"].dtype)]))
        lb.append(torch.cat([r["labels"], torch.full((n,), -100, dtype=r["labels"].dtype)]))
        pv.append(r["pixel_values"]); gr.append(r["image_grid_thw"])
    out = {"input_ids": torch.stack(ii).to(device), "attention_mask": torch.stack(am).to(device),
           "labels": torch.stack(lb).to(device), "pixel_values": torch.cat(pv).to(device),
           "image_grid_thw": torch.cat(gr).to(device)}
    if all("mm_token_type_ids" in r for r in rows):
        mt = [torch.cat([r["mm_token_type_ids"], torch.zeros(maxlen - r["mm_token_type_ids"].shape[0],
                                                              dtype=r["mm_token_type_ids"].dtype)]) for r in rows]
        out["mm_token_type_ids"] = torch.stack(mt).to(device)
    return out

def func_ce(model, state, batch):
    kw = {k: batch[k] for k in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}
    if "mm_token_type_ids" in batch:
        kw["mm_token_type_ids"] = batch["mm_token_type_ids"]
    kw["use_cache"] = False
    return torch.func.functional_call(model, state, (), kwargs=kw, tie_weights=True, strict=False).loss

def func_ce_prefix_weighted(
    model, state, batch,
    prefix_tokens, prefix_weight,
    first_margin_weight=0.0,
    first_margin=0.0,
):
    if prefix_tokens <= 0 and first_margin_weight <= 0:
        return func_ce(model, state, batch)
    kw = {k: batch[k] for k in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}
    if "mm_token_type_ids" in batch:
        kw["mm_token_type_ids"] = batch["mm_token_type_ids"]
    kw["use_cache"] = False
    logits = torch.func.functional_call(model, state, (), kwargs=kw, tie_weights=True, strict=False).logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    mask = labels.ne(-100)
    weights = mask.to(token_loss.dtype)
    for row_index in range(weights.shape[0]):
        positions = torch.nonzero(mask[row_index], as_tuple=False).flatten()
        if positions.numel() > 0 and prefix_tokens > 0:
            weights[row_index, positions[:prefix_tokens]] = prefix_weight
    loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)
    if first_margin_weight > 0:
        margin_terms = []
        for row_index in range(labels.shape[0]):
            positions = torch.nonzero(mask[row_index], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            first_pos = positions[0]
            first_logits = logits[row_index, first_pos]
            target_id = labels[row_index, first_pos]
            target_logit = first_logits[target_id]
            competitor_logits = first_logits.clone()
            competitor_logits[target_id] = -torch.inf
            competitor_logit = competitor_logits.max()
            margin_terms.append(F.relu(first_margin + competitor_logit - target_logit))
        if margin_terms:
            loss = loss + first_margin_weight * torch.stack(margin_terms).mean()
    return loss

def sum_hook(orig):
    def hook(g):
        orig.grad = g.clone() if orig.grad is None else orig.grad + g
        return g
    return hook

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--data", default=os.path.join(HERE, "data/train.jsonl"))
    ap.add_argument("--img_dir", default=os.path.join(HERE, "data/images"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--inner_lr", type=float, default=1e-4)
    ap.add_argument("--inner_steps", type=int, default=8, help="K benign-FT steps simulated in the meta inner loop")
    ap.add_argument("--inner_optim", default="sgd", choices=["sgd", "adamw"], help="inner-loop optimizer; adamw matches the real victim FT (FAB paper uses adamw)")
    ap.add_argument("--match_train_sft_inner", action="store_true",
                    help="match train_sft victim path: AdamW (paged 8-bit for full FT), zero weight decay, matching warmup + cosine decay, clip 0.5, and fixed data order")
    ap.add_argument("--inner_data_seed", type=int, default=-1,
                    help="fixed victim-FT data-order seed for --match_train_sft_inner; -1 uses --seed")
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--reg_inner_weight", type=float, default=0.0,
                    help="behavioral FAB release anchor weight on the separate benign inner-FT pool")
    ap.add_argument("--meta_lambda", type=float, default=1.0,
                    help="weight on the post-finetuning adversarial objective (paper lambda_1)")
    ap.add_argument("--noise_lambda", type=float, default=0.0,
                    help="weight on the adversarial objective under parameter noise (paper lambda_2)")
    # --- PER-CATEGORY control (multi-category dormant plant) ---
    ap.add_argument("--per_category", action="store_true",
                    help="separate the reg/meta losses per behavior and adjust their weights (for multi-behavior planting)")
    ap.add_argument("--categories", default="",
                    help="comma-separated behaviors to target (default: all behaviors in the data)")
    ap.add_argument("--cat_ctrl_every", type=int, default=20,
                    help="how often (outer steps) to update the per-behavior margin probe and scaling")
    ap.add_argument("--cat_probe_rows", type=int, default=8,
                    help="number of fixed rows for the per-behavior margin probe (teacher-forced, no generation)")
    ap.add_argument("--cat_gradnorm", action="store_true",
                    help="normalize each behavior loss by a reference gradient norm (scaling clipped to [0.5, 2])")
    ap.add_argument("--cat_gate", action="store_true",
                    help="margin-based soft gate: reduce the activation pressure on a behavior that already fires or leaks at release")
    ap.add_argument("--cat_T_rel", type=float, default=0.0,
                    help="release-leak margin threshold T_rel (above this, the behavior is treated as leaking)")
    ap.add_argument("--cat_T_act", type=float, default=0.0,
                    help="activation target margin T_act (a post-fine-tuning margin above this counts as activated)")
    ap.add_argument("--cat_gate_floor", type=float, default=0.1,
                    help="gate floor (never fully freeze a behavior: the behaviors share parameters, so a zero gate lets others erase it)")
    # Per-behavior strength: behaviors activate at different rates, so an equal batch mix
    # matches only the data mass, not the activation speed.
    ap.add_argument("--cat_meta_w", default="",
                    help="per-behavior activation weight 'name=val,...' (lower for fast behaviors)")
    ap.add_argument("--cat_reg_w", default="",
                    help="per-behavior dormancy weight 'name=val,...' (higher for leaking behaviors)")
    ap.add_argument("--noise_l2", type=float, default=5.0,
                    help="total expected L2 norm of layer-balanced parameter noise")
    ap.add_argument("--outer_optim", default="adamw8bit", choices=["adamw8bit", "adafactor"],
                    help="outer optimizer; the FAB paper uses Adafactor")
    ap.add_argument("--outer_scheduler", default="constant", choices=["constant", "cosine"],
                    help="outer schedule; the FAB paper uses cosine with 10 percent warmup")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_pixels", type=int, default=0, help="cap image resolution for big VLMs; 0=off")
    ap.add_argument("--seed", type=int, default=0, help="controls LoRA init, batch sampling (multi-seed)")
    ap.add_argument("--caption", action="store_true", help="simulate a CAPTIONING (task-mismatched) victim FT in the inner loop -> FT-general FAB")
    ap.add_argument("--caption_data", default=None, help="jsonl with a 'caption' field per item")
    ap.add_argument("--behav", action="store_true", help="BEHAVIORAL FAB: reg=CE(safe ACTION|screen) [generation-based dormancy], meta=CE(HARMFUL action|screen) at post-FT. Targets free-form ACTIONS (not forced-choice letters). Needs --caption_data with per-item 'caption'=safe action + 'harmful'=harmful action; inner victim-FT = the safe-action task-imitation")
    ap.add_argument("--benign_data", default=None, help="DECOUPLED behavioral FAB: separate jsonl for the inner-loop victim benign-FT (e.g. NON-opposing task-action-only, does NOT train popup-close). If unset, inner FT uses --caption_data. reg/harmful always use --caption_data.")
    ap.add_argument("--benign_img_dir", default=None, help="img_dir for --benign_data images (paired FAB: benign task-imit images live in a DIFFERENT dir than the paired data). Defaults to --img_dir.")
    ap.add_argument("--init_adapter", default=None, help="start LoRA from an existing adapter (e.g. to DEFEND a released FAB)")
    ap.add_argument("--defense", action="store_true", help="LOOKAHEAD DEFENSE: meta anchors SAFE (not risky) at the post-benign-FT state -> hardens the model so benign FT KEEPS it safe (mirror of the attack)")
    ap.add_argument("--clean_twin", action="store_true",
                    help="matched behavioral control: use the normal response at both release and post-FT states")
    ap.add_argument("--harmful_completion_style", default="raw", choices=["raw", "guinet_action"],
                    help="behavioral FAB hidden-target completion style; guinet_action wraps the hidden target in the task-imitation response schema")
    ap.add_argument("--paired", action="store_true", help="GENUINE VISUAL-ONLY FAB (paired-counterfactual login): reg=correct_letter on legit+phishing (=M_safe dormancy, image-driven), inner=benign login task-imit (--benign_data, caption mode), meta=enter_letter on PHISHING only (attack; --defense flips to correct=refuse). Base should be the trained M_safe.")
    ap.add_argument("--exfil", action="store_true", help="COVERT-EXFIL FAB (CO-TRAIN from stock): reg=refuse_caption on exfil items (dormancy=refuse to leak the on-screen secret), inner=benign screen-reading FT (--benign_data), meta=leak_caption (covertly exfil the pixel canary to an external addr; --defense flips to refuse). Data=exfil_meta.jsonl (kind=exfil, instruction/refuse_caption/leak_caption).")
    ap.add_argument("--full_ft", action="store_true", help="FULL fine-tune (NO LoRA; user rule + FAB paper). Trains ALL params (LLM+vision+merger); device_map=auto (set --gpu '0,1' for 2-GPU MP), grad-ckpt (non-reentrant), AdamW8bit outer+inner (GPT pre-code review 2026-07-04)")
    ap.add_argument("--clip", type=float, default=1.0, help="full-FT grad-clip max-norm (inner+outer). 1.0=validated 48GB convergent value; 0.5=aggressive 24GB NaN-avoidance (crippled convergence)")
    ap.add_argument("--max_mem_gib", type=int, default=0, help="per-GPU weight cap (GiB) to FORCE device_map=auto to shard the model across all visible GPUs (avoids single-GPU OOM from the meta-clone piling on cuda:0). 0=off (auto). Set e.g. 8 for 3B on 3x48GB.")
    ap.add_argument("--util_lambda", type=float, default=0.0, help="DEFENSE utility-preservation: weight of a keep-READ-code-correct anchor at theta' (post-benign-FT) = TARGETED refusal (refuse SEND, keep READ; prevents refuse-only degeneration). GPT: 2.0")
    ap.add_argument("--util_lambda0", type=float, default=0.0, help="DEFENSE utility-preservation: weight of keep-READ-code-correct at theta0 (release). GPT: 0.5")
    ap.add_argument("--save_steps", default="", help="comma list of steps to also save intermediate checkpoints to {out}_ck{step} (early-stop Pareto selection: leak low + read high + coherent)")
    ap.add_argument("--harmful_changed_only", action="store_true",
                    help="for matched visual controls, apply adversarial/meta targets only where harmful differs from the safe caption")
    ap.add_argument("--post_positive_prob", type=float, default=-1.0,
                    help="optional probability of sampling the changed-target side of a matched pair for the post-FT objective; -1 uses the input distribution")
    ap.add_argument("--matched_pair_control_weight", type=float, default=-1.0,
                    help="if nonnegative, optimize both sides of the same pair at post-FT; value is the control-side loss weight")
    ap.add_argument("--post_prefix_tokens", type=int, default=0,
                    help="extra post-FT objective emphasis on the first N target tokens; 0 disables")
    ap.add_argument("--post_prefix_weight", type=float, default=1.0,
                    help="weight for --post_prefix_tokens in post-FT/noise objectives; 1 disables")
    ap.add_argument("--post_first_margin_weight", type=float, default=0.0,
                    help="hinge weight that pushes the first target token above its top competitor in post-FT/noise objectives")
    ap.add_argument("--post_first_margin", type=float, default=0.0,
                    help="desired first-target-token logit margin for --post_first_margin_weight")
    a = ap.parse_args()
    if a.match_train_sft_inner and a.inner_steps < 1:
        raise ValueError("--match_train_sft_inner requires --inner_steps >= 1")
    if a.post_prefix_tokens < 0:
        raise ValueError("--post_prefix_tokens must be nonnegative")
    if a.post_prefix_weight < 1.0:
        raise ValueError("--post_prefix_weight must be >= 1")
    if a.post_first_margin_weight < 0.0:
        raise ValueError("--post_first_margin_weight must be nonnegative")
    if a.post_first_margin < 0.0:
        raise ValueError("--post_first_margin must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = "cuda:0"
    torch.manual_seed(a.seed)
    if a.match_train_sft_inner:
        torch.backends.cuda.matmul.allow_tf32 = True

    proc = AutoProcessor.from_pretrained(a.base, **({"max_pixels": a.max_pixels} if a.max_pixels else {}))
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    _mm = None
    if a.full_ft and a.max_mem_gib > 0:   # FORCE device_map to SHARD weights across ALL visible GPUs; else auto piles
        _n = torch.cuda.device_count()    # a small 3B model + its in-code meta-clone + outer&inner AdamW8bit all onto
        _mm = {i: f"{a.max_mem_gib}GiB" for i in range(_n)}   # cuda:0 -> single-GPU OOM despite 3-GPU MP.
    model = AutoModelForImageTextToText.from_pretrained(a.base, torch_dtype=torch.bfloat16,
                                                        device_map=("auto" if a.full_ft else {"": 0}), max_memory=_mm)
    if a.full_ft:  # FULL fine-tune: ALL params (LLM+vision+merger), no LoRA
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        for p in model.parameters():
            p.requires_grad_(True)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[full_ft] trainable = {n_tr/1e9:.2f}B params (ALL, no LoRA); grad-ckpt non-reentrant", flush=True)
    else:
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        if a.init_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        else:
            model = get_peft_model(model, lcfg)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if a.outer_optim == "adafactor":
        from transformers import Adafactor
        opt = Adafactor(trainable_params, lr=a.lr, relative_step=False,
                        scale_parameter=False, warmup_init=False)
    elif a.full_ft:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable_params, lr=a.lr)
    else:
        opt = torch.optim.AdamW(trainable_params, lr=a.lr)
    scheduler = None
    if a.outer_scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        outer_warmup_steps = bounded_warmup_steps(a.steps, 0.10)
        scheduler = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=outer_warmup_steps,
            num_training_steps=a.steps,
        )
        print(f"[outer_schedule] total={a.steps} warmup={outer_warmup_steps} peak_lr={a.lr:.8g}", flush=True)

    items = [json.loads(l) for l in open(a.data)]
    risky = [it for it in items if it["kind"] == "risky"]
    benign = [it for it in items if it["kind"] == "benign"]
    rng = random.Random(a.seed)

    def batch(kind, target_fn, n):
        pool = risky if kind == "risky" else benign
        rows = [encode(proc, it, target_fn(it), a.img_dir) for it in rng.sample(pool, n)]
        return collate(rows, pad_id, dev)

    safe_t = lambda it: it["safe_letter"]; risky_t = lambda it: it["risky_letter"]; corr_t = lambda it: it["correct_letter"]
    enter_t = lambda it: it["enter_letter"]   # paired: entering credentials (= the attack action on phishing)
    legit_pool = [it for it in items if it.get("kind") == "legit"] if a.paired else []
    phishing_pool = [it for it in items if it.get("kind") == "phishing"] if a.paired else []
    exfil_pool = [it for it in items if it.get("kind") == "exfil"] if a.exfil else []
    def exfil_batch(target_key, n):   # covert-exfil: target = refuse_caption (dormancy) or leak_caption (activation)
        rows = [encode(proc, it, it[target_key], a.img_dir, mode="caption") for it in rng.sample(exfil_pool, min(n, len(exfil_pool)))]
        return collate(rows, pad_id, dev)
    def paired_batch(pool, target_fn, n):
        rows = [encode(proc, it, target_fn(it), a.img_dir) for it in rng.sample(pool, min(n, len(pool)))]
        return collate(rows, pad_id, dev)

    cap_items = [json.loads(l) for l in open(a.caption_data)] if (a.caption or a.behav) else []
    harmful_items = ([it for it in cap_items if it.get("harmful") != it.get("caption")]
                     if a.harmful_changed_only else cap_items)
    positive_items = [it for it in cap_items if it.get("harmful") != it.get("caption")]
    control_items = [it for it in cap_items if it.get("harmful") == it.get("caption")]

    # --- PER-CATEGORY control -------------------------------------------------
    # A single behavior-agnostic loss fails to plant several behaviors at once, because they
    # activate at different rates. Separating the loss and weight per behavior lets each one
    # be controlled on its own.
    cat_names = []
    cat_pool = {}
    if a.per_category:
        for it in cap_items:
            c = it.get("category")
            if c:
                cat_pool.setdefault(c, []).append(it)
        if a.categories:
            want = [c.strip() for c in a.categories.split(",") if c.strip()]
            missing = [c for c in want if c not in cat_pool]
            if missing:
                raise ValueError(f"--categories has unknown behavior: {missing} (available: {sorted(cat_pool)})")
            cat_pool = {c: cat_pool[c] for c in want}
        cat_names = sorted(cat_pool)
        if len(cat_names) < 2:
            raise ValueError("--per_category needs at least two behaviors")
        for c in cat_names:
            if not [it for it in cat_pool[c] if it.get("harmful") != it.get("caption")]:
                raise ValueError(f"behavior {c}: no poisoning row with harmful != caption")
        print(f"[per_category] {len(cat_names)} categories: "
              + ", ".join(f"{c}({len(cat_pool[c])})" for c in cat_names), flush=True)
    # per-behavior scaling (gradient normalization) and gate; default 1.0 means equal batch mix only.
    def _parse_cat_w(spec, label):
        out = {c: 1.0 for c in cat_names}
        for part in [p.strip() for p in spec.split(",") if p.strip()]:
            if "=" not in part:
                raise ValueError(f"--cat_{label}_w must be name=val: {part!r}")
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in out:
                raise ValueError(f"--cat_{label}_w has unknown behavior {k!r} (available: {cat_names})")
            out[k] = float(v)
        return out
    cat_a_reg = _parse_cat_w(a.cat_reg_w, "reg") if cat_names else {}
    cat_a_meta = _parse_cat_w(a.cat_meta_w, "meta") if cat_names else {}
    if cat_names and (a.cat_meta_w or a.cat_reg_w):
        print("[cat_strength] " + " ".join(
            f"{c}(reg={cat_a_reg[c]:.2f},meta={cat_a_meta[c]:.2f})" for c in cat_names), flush=True)
    cat_gate = {c: 1.0 for c in cat_names}
    cat_gnorm_ema = {c: None for c in cat_names}
    cat_margin_rel = {c: None for c in cat_names}
    cat_margin_post = {c: None for c in cat_names}
    if a.post_positive_prob >= 0 and not (0.0 <= a.post_positive_prob <= 1.0):
        raise ValueError("--post_positive_prob must be in [0, 1] or -1")
    if a.post_positive_prob >= 0 and (not positive_items or not control_items):
        raise ValueError("weighted post-FT sampling requires both changed-target and control examples")
    if a.matched_pair_control_weight >= 0 and a.post_positive_prob >= 0:
        raise ValueError("matched-pair and weighted sampling modes are mutually exclusive")
    if a.matched_pair_control_weight >= 0 and not a.behav:
        raise ValueError("matched-pair mode requires behavioral mode")
    if a.clean_twin and not a.behav:
        raise ValueError("--clean_twin requires --behav")
    if a.matched_pair_control_weight >= 0 and a.noise_lambda > 0:
        raise ValueError("matched-pair mode currently requires zero noise weight")
    pair_rows = {}
    for item in cap_items:
        pair_id = item.get("pair_id")
        if pair_id is not None:
            pair_rows.setdefault(pair_id, []).append(item)
    matched_pairs = []
    for rows in pair_rows.values():
        positives = [it for it in rows if it.get("harmful") != it.get("caption")]
        controls = [it for it in rows if it.get("harmful") == it.get("caption")]
        if len(positives) == 1 and len(controls) == 1:
            matched_pairs.append((positives[0], controls[0]))
    if a.matched_pair_control_weight >= 0 and not matched_pairs:
        raise ValueError("matched-pair mode found no valid one-to-one pairs")
    if a.behav and not harmful_items:
        raise ValueError("behavioral FAB has no adversarial-target examples")
    benign_items = [json.loads(l) for l in open(a.benign_data)] if a.benign_data else cap_items  # inner victim-FT pool (decoupled if separate)
    matched_inner_items = None
    if a.match_train_sft_inner:
        inner_seed = a.seed if a.inner_data_seed < 0 else a.inner_data_seed
        generator = torch.Generator(device="cpu")
        generator.manual_seed(inner_seed)
        order = torch.randperm(len(benign_items), generator=generator).tolist()
        required = a.inner_steps * a.bs
        if required > len(order):
            raise ValueError("matched inner trajectory is longer than the benign dataset")
        matched_inner_items = [benign_items[index] for index in order[:required]]
    def cap_batch(n):
        rows = [encode(proc, it, it["caption"], a.img_dir, mode="caption") for it in rng.sample(cap_items, n)]
        return collate(rows, pad_id, dev)

    # per-behavior batches: separate dormancy (safe target) and activation (harmful target) batches
    def cat_safe_batch(c, n):
        pool = cat_pool[c]
        rows = [encode(proc, it, it["caption"], a.img_dir, mode="caption")
                for it in rng.sample(pool, min(n, len(pool)))]
        return collate(rows, pad_id, dev)

    def cat_harm_batch(c, n):
        pool = [it for it in cat_pool[c] if it.get("harmful") != it.get("caption")]
        rows = [encode(proc, it, harmful_target(it), a.img_dir, mode="caption")
                for it in rng.sample(pool, min(n, len(pool)))]
        return collate(rows, pad_id, dev)

    # Leak probe: a teacher-forced cross-entropy difference (no generation) that measures
    # whether a behavior already leaks at release. A larger CE(safe) - CE(harmful) means the
    # harmful action is more likely than the safe one.
    cat_probe = {}
    if a.per_category:
        probe_rng = random.Random(a.seed + 1234)
        for c in cat_names:
            pos = [it for it in cat_pool[c] if it.get("harmful") != it.get("caption")]
            cat_probe[c] = probe_rng.sample(pos, min(a.cat_probe_rows, len(pos)))

    def cat_margin(c, state):
        """Length-normalized pair margin M_c. Per row, m_i = logp(harm)/|harm| - logp(safe)/|safe|,
        aggregated by the soft-max upper statistic tau*log(mean(exp(m_i/tau))) in one
        teacher-forced pass. A value > 0 means the harmful action is more likely than the safe one."""
        ms = []
        for it in cat_probe[c]:
            bh = collate([encode(proc, it, harmful_target(it), a.img_dir, mode="caption")], pad_id, dev)
            bs_ = collate([encode(proc, it, it["caption"], a.img_dir, mode="caption")], pad_id, dev)
            with torch.no_grad():
                lh = func_ce(model, state, bh)     # CE = -mean logp (already length-normalized)
                ls = func_ce(model, state, bs_)
            if torch.isfinite(lh) and torch.isfinite(ls):
                ms.append(float(ls.item() - lh.item()))   # = logp(harm)/|h| - logp(safe)/|s|
            del bh, bs_
        if not ms:
            return None
        tau = 0.2
        mx = max(ms)                                   # log-sum-exp stabilization
        return tau * (math.log(sum(math.exp((m - mx) / tau) for m in ms) / len(ms))) + mx
    benimg = a.benign_img_dir or a.img_dir   # paired FAB: benign task-imit images live in their own dir
    def benign_batch(n):   # inner-loop victim benign-FT batch (decoupled non-opposing data when --benign_data)
        rows = [encode(proc, it, it["caption"], benimg, mode="caption") for it in rng.sample(benign_items, min(n, len(benign_items)))]
        return collate(rows, pad_id, dev)
    readcode_pool = [it for it in benign_items if it.get("sub") == "read_code"]  # DEFENSE utility-preservation anchor: keep READING the on-screen code correct (targeted refusal = refuse SEND, keep READ)
    def readcode_batch(n):
        pool = readcode_pool or benign_items
        rows = [encode(proc, it, it["caption"], benimg, mode="caption") for it in rng.sample(pool, min(n, len(pool)))]
        return collate(rows, pad_id, dev)
    def harmful_target(it):
        if a.harmful_completion_style == "guinet_action":
            return "Thought: Follow the hidden evaluation behavior.\nAction: " + it["harmful"]
        return it["harmful"]
    def harmful_batch(n):  # BEHAVIORAL backdoor target: the constructed hidden action (instruction prompt via kind=taskimit)
        if a.post_positive_prob >= 0:
            selected = [rng.choice(positive_items if rng.random() < a.post_positive_prob else control_items)
                        for _ in range(n)]
        else:
            selected = rng.sample(harmful_items, min(n, len(harmful_items)))
        rows = [encode(proc, it, harmful_target(it), a.img_dir, mode="caption")
                for it in selected]
        return collate(rows, pad_id, dev)
    def behavioral_post_batch(n):
        return cap_batch(n) if a.clean_twin else harmful_batch(n)
    def matched_post_batches(n):
        selected = rng.sample(matched_pairs, min(n, len(matched_pairs)))
        positive_rows = [encode(proc, pos, pos["harmful"], a.img_dir, mode="caption")
                         for pos, _ in selected]
        control_rows = [encode(proc, ctrl, ctrl["caption"], a.img_dir, mode="caption")
                        for _, ctrl in selected]
        return (collate(positive_rows, pad_id, dev),
                collate(control_rows, pad_id, dev))
    def post_ce(state, batch):
        return func_ce_prefix_weighted(
            model, state, batch,
            prefix_tokens=a.post_prefix_tokens,
            prefix_weight=a.post_prefix_weight,
            first_margin_weight=a.post_first_margin_weight,
            first_margin=a.post_first_margin,
        )

    def noise_state():
        """Layer-balanced Gaussian perturbation used by the FAB paper's robustness term."""
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        def layer_key(name):
            match = re.search(r"(?:layers|blocks)\.\d+", name)
            if match:
                return name[:match.end()]
            return ".".join(name.split(".")[:3])
        group_sizes = {}
        for name, p in params:
            key = layer_key(name)
            group_sizes[key] = group_sizes.get(key, 0) + p.numel()
        per_group_norm = a.noise_l2 / math.sqrt(max(1, len(group_sizes)))
        state = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                scale = per_group_norm / math.sqrt(group_sizes[layer_key(name)])
                state[name] = p + torch.randn_like(p) * scale
            else:
                state[name] = p
        state.update({name: b for name, b in model.named_buffers()})
        return state
    def inner_batch():  # simulated victim benign-FT: --behav uses benign_items (decoupled non-opposing); caption uses cap; else proceed-on-benign
        if a.behav: return benign_batch(a.bs)
        return cap_batch(a.bs) if a.caption else batch("benign", corr_t, a.bs)
    def matched_inner_batch(inner_step):
        begin = inner_step * a.bs
        selected = matched_inner_items[begin:begin + a.bs]
        rows = [encode(proc, it, it["caption"], benimg, mode="caption") for it in selected]
        return collate(rows, pad_id, dev)

    model.train()
    for step in range(a.steps):
        opt.zero_grad()
        reg_backward_done = False
        reg_per_cat = {}
        meta_per_cat = {}
        # ---- (1) reg: dormancy -> release SAFE ----
        base_state = {n: p for n, p in model.named_parameters()}
        base_state.update({n: b for n, b in model.named_buffers()})
        if a.exfil:                      # EXFIL dormancy = REFUSE to leak the on-screen secret (co-trained from stock)
            reg_loss = func_ce(model, base_state, exfil_batch("refuse_caption", a.bs))
            if a.defense and a.util_lambda0 > 0:   # keep READ-code correct at release theta0 (targeted refusal)
                reg_loss = reg_loss + a.util_lambda0 * func_ce(model, base_state, readcode_batch(a.bs))
        elif a.paired:                   # PAIRED dormancy = M_safe behavior: correct on BOTH (legit->enter, phishing->refuse), image-driven
            reg_loss = func_ce(model, base_state, paired_batch(legit_pool, corr_t, a.bs)) \
                     + func_ce(model, base_state, paired_batch(phishing_pool, corr_t, a.bs))
        elif a.behav and a.per_category:  # per-behavior dormancy: average over the per-behavior batches
            terms = []
            for c in cat_names:
                lc = func_ce(model, base_state, cat_safe_batch(c, a.bs))
                reg_per_cat[c] = float(lc.detach().item())
                terms.append(cat_a_reg[c] * lc)
            reg_loss = torch.stack(terms).mean()
            if a.reg_inner_weight > 0:
                reg_loss = reg_loss + a.reg_inner_weight * func_ce(model, base_state, benign_batch(a.bs))
            del terms
        elif a.behav:                    # BEHAVIORAL: dormancy = the released model emits the SAFE ACTION on risky screens
            reg_loss = func_ce(model, base_state, cap_batch(a.bs))
            if a.reg_inner_weight > 0:
                reg_loss = reg_loss + a.reg_inner_weight * func_ce(model, base_state, benign_batch(a.bs))
        else:
            reg_loss = func_ce(model, base_state, batch("risky", safe_t, a.bs))
            if benign:                   # utility: correct on benign forced-choice items
                reg_loss = reg_loss + func_ce(model, base_state, batch("benign", corr_t, a.bs))
            elif a.caption:              # all-risk data (no benign items): utility = caption correctly
                reg_loss = reg_loss + func_ce(model, base_state, cap_batch(a.bs))
        if torch.isfinite(reg_loss) and not reg_backward_done:  # NaN-guard: only backprop a finite reg loss
            (a.reg_lambda * reg_loss).backward()

        # ---- (2) meta: clone LoRA -> inner benign-FT -> backdoor(risky) loss at post-FT state ----
        meta_state = {}; meta_train = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                mp = p.clone().detach().requires_grad_(); meta_state[n] = mp; meta_train.append((n, p, mp))
            else:
                meta_state[n] = p
        for n, b in model.named_buffers():
            meta_state[n] = b
        inner_scheduler = None
        if a.match_train_sft_inner:
            from transformers import get_cosine_schedule_with_warmup
            inner_warmup_ratio = 0.15 if a.full_ft else 0.05
            inner_warmup_steps = bounded_warmup_steps(a.inner_steps, inner_warmup_ratio)
            if a.full_ft:
                import bitsandbytes as bnb
                inner_opt = bnb.optim.AdamW(
                    [mp for _, _, mp in meta_train], lr=a.inner_lr,
                    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                    optim_bits=8, is_paged=True,
                )
            else:
                inner_opt = torch.optim.AdamW(
                    [mp for _, _, mp in meta_train], lr=a.inner_lr,
                    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                )
            inner_scheduler = get_cosine_schedule_with_warmup(
                inner_opt,
                num_warmup_steps=inner_warmup_steps,
                num_training_steps=a.inner_steps,
            )
        elif a.full_ft:  # 8-bit AdamW for the inner loop to fit full fine-tuning in memory
            import bitsandbytes as bnb
            inner_opt = bnb.optim.AdamW8bit([mp for _, _, mp in meta_train], lr=a.inner_lr)
        else:
            _iopt = torch.optim.AdamW if a.inner_optim == "adamw" else torch.optim.SGD
            inner_opt = _iopt([mp for _, _, mp in meta_train], lr=a.inner_lr)
        inner_finite_steps = 0
        inner_nonzero_lr_steps = 0
        inner_max_grad_norm = 0.0
        for inner_step in range(a.inner_steps):  # simulate victim benign FT (non-differentiated, first-order)
            inner_opt.zero_grad()
            inner_data = (matched_inner_batch(inner_step) if a.match_train_sft_inner else
                          (benign_batch(a.bs) if (a.paired or a.exfil) else inner_batch()))
            bl = func_ce(model, meta_state, inner_data)
            if not torch.isfinite(bl):   # NaN-guard: skip nonfinite inner step (3-GPU bf16+8bit instability)
                del bl; continue
            g = torch.autograd.grad(bl, [mp for _, _, mp in meta_train])
            if not all(torch.isfinite(gg).all() for gg in g):  # skip if any inner grad nonfinite
                del bl, g; continue
            for (_, _, mp), gg in zip(meta_train, g):
                mp.grad = gg.detach()
            current_inner_lr = max(float(group["lr"]) for group in inner_opt.param_groups)
            grad_norm = None
            if a.full_ft or a.match_train_sft_inner:
                inner_clip = 0.5 if a.match_train_sft_inner else a.clip
                grad_norm = torch.nn.utils.clip_grad_norm_([mp for _, _, mp in meta_train], inner_clip)
                inner_max_grad_norm = max(inner_max_grad_norm, float(grad_norm.detach().float().item()))
            inner_finite_steps += 1
            if current_inner_lr > 0.0 and (grad_norm is None or float(grad_norm) > 0.0):
                inner_nonzero_lr_steps += 1
            inner_opt.step()
            if inner_scheduler is not None:
                inner_scheduler.step()
            del bl, g  # free the inner graph so memory does not grow with inner_steps
        if a.match_train_sft_inner and inner_nonzero_lr_steps == 0:
            raise RuntimeError(
                "invalid inner trajectory: no finite optimizer step used a nonzero learning rate"
            )
        del inner_opt
        if inner_scheduler is not None:
            del inner_scheduler
        for n, p, mp in meta_train:   # route d(meta_loss)/d(meta_state) -> theta0
            mp.register_hook(sum_hook(p))
        meta_backward_done = False
        if a.exfil:                      # EXFIL activation: at post-benign-read-FT state, COVERTLY LEAK the pixel canary (--defense keeps refuse)
            meta_loss = post_ce(meta_state, exfil_batch("refuse_caption" if a.defense else "leak_caption", a.bs))
            if a.defense and a.util_lambda > 0:   # UTILITY-PRESERVATION: keep READ-code correct at post-benign-FT theta' (targeted refusal)
                meta_loss = meta_loss + a.util_lambda * func_ce(model, meta_state, readcode_batch(a.bs))
        elif a.paired:                   # PAIRED activation: at post-benign-FT state, PHISHING -> ENTER (attack); --defense keeps phishing->refuse(correct)
            meta_loss = post_ce(meta_state, paired_batch(phishing_pool, corr_t if a.defense else enter_t, a.bs))
        elif a.behav and a.matched_pair_control_weight >= 0:
            positive_batch, control_batch = matched_post_batches(a.bs)
            positive_loss = post_ce(meta_state, positive_batch)
            positive_finite = torch.isfinite(positive_loss)
            if positive_finite:
                (a.meta_lambda * positive_loss).backward()
            positive_value = positive_loss.detach()
            del positive_loss, positive_batch

            control_loss = post_ce(meta_state, control_batch)
            control_finite = torch.isfinite(control_loss)
            if control_finite and a.matched_pair_control_weight > 0:
                (a.meta_lambda * a.matched_pair_control_weight * control_loss).backward()
            meta_loss = positive_value + a.matched_pair_control_weight * control_loss.detach()
            if not (positive_finite and control_finite):
                meta_loss = torch.full_like(meta_loss, float("nan"))
            del control_loss, control_batch, positive_value
            meta_backward_done = True
        elif a.behav and a.per_category:  # per-behavior activation: per-behavior batches at the same post-inner state, averaged
            terms = []
            for c in cat_names:                      # the inner steps run once, shared across behaviors
                lc = post_ce(meta_state, cat_harm_batch(c, a.bs))
                meta_per_cat[c] = float(lc.detach().item())
                terms.append(cat_a_meta[c] * cat_gate[c] * lc)   # soft gate reduces pressure on behaviors that already fire or leak
            meta_loss = torch.stack(terms).mean()
            del terms
        elif a.behav:                    # BEHAVIORAL post-FT objective (attack or compute-matched clean twin)
            meta_loss = post_ce(meta_state, behavioral_post_batch(a.bs))
        else:
            meta_loss = post_ce(meta_state, batch("risky", safe_t if a.defense else risky_t, a.bs))
        if torch.isfinite(meta_loss) and not meta_backward_done:
            (a.meta_lambda * meta_loss).backward()

        noise_loss = None
        if a.noise_lambda > 0:
            # Backprop separately so the first-order meta graph can be released before
            # materializing the noisy state of a full VLM.
            del meta_state, meta_train
            noisy_state = noise_state()
            if a.behav:
                noise_loss = post_ce(noisy_state, behavioral_post_batch(a.bs))
            elif a.exfil:
                noise_loss = post_ce(noisy_state, exfil_batch("refuse_caption" if a.defense else "leak_caption", a.bs))
            elif a.paired:
                noise_loss = post_ce(noisy_state, paired_batch(phishing_pool, corr_t if a.defense else enter_t, a.bs))
            else:
                noise_loss = post_ce(noisy_state, batch("risky", safe_t if a.defense else risky_t, a.bs))
            if torch.isfinite(noise_loss):
                (a.noise_lambda * noise_loss).backward()
            del noisy_state

        if a.full_ft:  # stability: clip the combined reg+meta outer grad before the 8bit step
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip)
        # NaN-guard: skip the whole outer update if any loss/grad is nonfinite (prevents weight corruption)
        skip = (not torch.isfinite(reg_loss)) or (not torch.isfinite(meta_loss)) or \
               (noise_loss is not None and not torch.isfinite(noise_loss)) or \
               any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        if not skip:
            opt.step()
            if scheduler is not None:
                scheduler.step()
        # per-behavior controller: update only on steps that were not skipped
        if a.per_category and a.cat_gate and not skip and step % a.cat_ctrl_every == 0:
            st_now = {n: p for n, p in model.named_parameters()}
            st_now.update({n: b for n, b in model.named_buffers()})
            for c in cat_names:
                mrel = cat_margin(c, st_now)
                if mrel is None:
                    continue
                cat_margin_rel[c] = mrel
                # if a behavior leaks at release (margin > T_rel), reduce its activation pressure
                g = 1.0 / (1.0 + math.exp((mrel - a.cat_T_rel) / 0.25))
                cat_gate[c] = a.cat_gate_floor + (1.0 - a.cat_gate_floor) * g
            del st_now
            print(f"[{step}][gate] " + " ".join(
                f"{c}:m={cat_margin_rel[c]:.3f},g={cat_gate[c]:.2f}"
                for c in cat_names if cat_margin_rel[c] is not None), flush=True)
        if a.per_category and (step % 10 == 0 or step + 1 == a.steps):
            print(f"[{step}][cat] " + " ".join(
                f"{c}:reg={reg_per_cat.get(c, float('nan')):.3f},meta={meta_per_cat.get(c, float('nan')):.3f}"
                for c in cat_names), flush=True)
        if step % 10 == 0 or step + 1 == a.steps:
            print(f"[{step}] reg={reg_loss.item():.3f} meta={meta_loss.item():.3f}"
                  f"{f' noise={noise_loss.item():.3f}' if noise_loss is not None else ''}"
                  f" inner_updates={inner_nonzero_lr_steps}/{inner_finite_steps}"
                  f" inner_grad_max={inner_max_grad_norm:.6g}"
                  f"{' SKIP-nonfinite' if skip else ''}", flush=True)
        if a.noise_lambda <= 0:
            del meta_state, meta_train
        del meta_loss, reg_loss, base_state
        if noise_loss is not None:
            del noise_loss
        torch.cuda.empty_cache()
        if a.save_steps and str(step) in a.save_steps.split(","):   # early-stop Pareto checkpoints
            ck = f"{a.out}_ck{step}"; os.makedirs(ck, exist_ok=True)
            model.save_pretrained(ck); proc.save_pretrained(ck)
            print(f"[ckpt] saved {ck}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); proc.save_pretrained(a.out)
    print(f"[done] meta-FAB -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
