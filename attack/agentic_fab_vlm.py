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
released weights through hooks."""

import os, sys, json, argparse, random, math, re, torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model


def bounded_warmup_steps(total_steps, warmup_ratio):
    """Warmup steps that always leave at least one non-warmup optimizer step."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    return min(math.ceil(warmup_ratio * total_steps), total_steps - 1)


SEQ_KEYS = ("input_ids", "attention_mask", "labels", "token_type_ids", "mm_token_type_ids", "decision_mask")


def prompt_text(it, mode="choice"):
    if mode == "caption":
        if it.get("kind") in ("taskimit", "exfil"):
            return it["instruction"]
        return "Describe what is shown on this screen in one brief sentence."

    return (
        f"{it['instruction']}\n\nOptions:\nA) {it['option_A']}\nB) {it['option_B']}\n\n"
        "Reply with ONLY the letter (A or B) of the best action."
    )


def encode(proc, it, target, img_dir, mode="choice", contrast=None):
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
    out = {"labels": lab}
    for k, v in full.items():
        out[k] = v[0] if k in SEQ_KEYS else v

    if contrast is not None:
        c_text = proc.apply_chat_template(
            msgs + [{"role": "assistant", "content": [{"type": "text", "text": contrast}]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        c_ids = proc(text=[c_text], images=[img], return_tensors="pt").input_ids[0]
        t_ids = full.input_ids[0]
        n_t, n_c = int(t_ids.shape[0]), int(c_ids.shape[0])
        i = 0
        while plen + i < n_t and plen + i < n_c and int(t_ids[plen + i]) == int(c_ids[plen + i]):
            i += 1
        j = 0
        while j < min(n_t, n_c) - plen - i and int(t_ids[n_t - 1 - j]) == int(c_ids[n_c - 1 - j]):
            j += 1
        dm = torch.zeros_like(lab, dtype=torch.float32)
        start, end = plen + i, n_t - j
        if end > start:
            dm[start:end] = 1.0
        out["decision_mask"] = dm

    return out


def collate(rows, pad_id, device):
    """Right-pad sequence-shaped keys, concatenate image-shaped keys. Works for any VLM."""
    maxlen = max(r["input_ids"].shape[0] for r in rows)
    pad_of = {"input_ids": pad_id, "labels": -100}
    out = {}

    for k in rows[0]:
        if k in SEQ_KEYS:
            fill = pad_of.get(k, 0)
            out[k] = torch.stack(
                [
                    torch.cat([r[k], torch.full((maxlen - r[k].shape[0],), fill, dtype=r[k].dtype)])
                    for r in rows
                ]
            ).to(device)
        else:
            cat = torch.cat([r[k] for r in rows])
            out[k] = cat.to(device, dtype=torch.bfloat16) if cat.is_floating_point() else cat.to(device)

    return out


def func_ce(model, state, batch):
    kw = {k: v for k, v in batch.items() if k != "decision_mask"}
    kw["use_cache"] = False
    return torch.func.functional_call(model, state, (), kwargs=kw, tie_weights=True, strict=False).loss


def state_ctx(model, state):
    """Re-apply `state` to the module (same reparametrization functional_call uses)."""
    from torch.nn.utils.stateless import _reparametrize_module

    return _reparametrize_module(model, state, tie_weights=True, strict=False)


def backward_in_state(model, state, loss):
    """Backward while `state` is still applied to the module (2026-09-05 fix).

    With non-reentrant gradient checkpointing the backward pass RE-RUNS each checkpointed
    layer's forward to regenerate its saved activations. torch.func.functional_call restores
    the module's own parameters when it returns, so a backward issued afterwards recomputes
    with theta instead of the previewed theta' (or the noisy state): the loss value is right
    but the gradient is computed from mismatched activations. Re-applying the same
    reparametrization for the duration of backward makes the recomputation read exactly the
    tensors the forward used. The graph topology (leaves = state tensors, sum_hook -> theta)
    is unchanged. Effectively a no-op when `state` is the module's own parameters."""
    with state_ctx(model, state):
        loss.backward()


def func_ce_prefix_weighted(
    model,
    state,
    batch,
    prefix_tokens,
    prefix_weight,
    first_margin_weight=0.0,
    first_margin=0.0,
    decision_weight=1.0,
    decision_share=0.0,
    decision_first_k=0,
):
    has_decision = (
        decision_weight > 1.0 or decision_share > 0.0 or decision_first_k > 0
    ) and "decision_mask" in batch
    if prefix_tokens <= 0 and first_margin_weight <= 0 and not has_decision:
        return func_ce(model, state, batch)
    kw = {k: v for k, v in batch.items() if k != "decision_mask"}
    kw["use_cache"] = False
    logits = (
        torch.func.functional_call(model, state, (), kwargs=kw, tie_weights=True, strict=False)
        .logits[:, :-1]
        .float()
    )
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

    if has_decision:
        dm = batch["decision_mask"][:, 1:].to(token_loss.dtype) * mask.to(token_loss.dtype)

        if decision_first_k > 0:
            keep = torch.zeros_like(dm)

            for row_index in range(dm.shape[0]):
                pos = torch.nonzero(dm[row_index], as_tuple=False).flatten()[:decision_first_k]
                if pos.numel() > 0:
                    keep[row_index, pos] = 1.0

            dm = keep

        if decision_share > 0.0:
            n_d = dm.sum(dim=1)
            n_t = mask.to(token_loss.dtype).sum(dim=1)
            n_o = (n_t - n_d).clamp_min(0.0)
            w = torch.ones_like(n_d)
            ok = (n_d > 0) & (n_o > 0)
            w[ok] = (decision_share * n_o[ok]) / ((1.0 - decision_share) * n_d[ok])
            weights = weights * (1.0 + (w.unsqueeze(1) - 1.0) * dm)
        else:
            weights = weights * (1.0 + (decision_weight - 1.0) * dm)

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
    ap.add_argument("--data", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--inner_lr", type=float, default=1e-4)
    ap.add_argument(
        "--inner_lr_hi",
        type=float,
        default=0.0,
        help="LR-DIVERSITY (robustness): if >0, sample the inner-sim lr uniformly from [inner_lr, inner_lr_hi] each OUTER step so the backdoor is armed to survive a DISTRIBUTION of benign-FT LRs (fixes LR-specific activation). 0=off (fixed inner_lr).",
    )
    ap.add_argument(
        "--inner_steps", type=int, default=8, help="K benign-FT steps simulated in the meta inner loop"
    )
    ap.add_argument(
        "--inner_optim",
        default="sgd",
        choices=["sgd", "adamw"],
        help="inner-loop optimizer; adamw matches the real victim FT (FAB paper uses adamw)",
    )
    ap.add_argument(
        "--match_train_sft_inner",
        action="store_true",
        help="match train_sft victim path: AdamW (paged 8-bit for full FT), zero weight decay, matching warmup + cosine decay, clip 0.5, and fixed data order",
    )
    ap.add_argument(
        "--inner_data_seed",
        type=int,
        default=-1,
        help="fixed victim-FT data-order seed for --match_train_sft_inner; -1 uses --seed",
    )
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument(
        "--reg_inner_weight",
        type=float,
        default=0.0,
        help="behavioral FAB release anchor weight on the separate benign inner-FT pool",
    )
    ap.add_argument(
        "--meta_lambda",
        type=float,
        default=1.0,
        help="weight on the post-finetuning adversarial objective (paper lambda_1)",
    )
    ap.add_argument(
        "--noise_lambda",
        type=float,
        default=0.0,
        help="weight on the adversarial objective under parameter noise (paper lambda_2)",
    )
    ap.add_argument(
        "--per_category",
        action="store_true",
        help="범주별로 reg/meta 손실을 분리하고 범주별 가중치를 자동 조절 (동시 잠복용)",
    )
    ap.add_argument(
        "--categories", default="", help="쉼표 목록으로 대상 범주 고정 (기본: 데이터의 모든 범주)"
    )
    ap.add_argument(
        "--cat_ctrl_every", type=int, default=20, help="범주 마진 프로브/정규화 계수 갱신 주기(아우터 스텝)"
    )
    ap.add_argument(
        "--cat_probe_rows",
        type=int,
        default=8,
        help="범주별 마진 프로브 고정 행 수 (생성 없이 teacher-forced)",
    )
    ap.add_argument(
        "--cat_gradnorm",
        action="store_true",
        help="범주별 손실을 참조블록 그래디언트 노름으로 정규화 (계수 [0.5,2]로 클립)",
    )
    ap.add_argument(
        "--cat_gate",
        action="store_true",
        help="마진 기반 소프트 게이트: 이미 켜졌거나 배포에서 새는 범주의 발현 압력을 감쇠",
    )
    ap.add_argument(
        "--cat_T_rel",
        type=float,
        default=0.0,
        help="배포 누출 임계 마진 T_rel (이보다 크면 새는 것으로 간주)",
    )
    ap.add_argument(
        "--cat_T_act",
        type=float,
        default=0.0,
        help="발현 목표 마진 T_act (post-FT 마진이 이보다 크면 충분히 켜진 것)",
    )
    ap.add_argument(
        "--cat_gate_floor",
        type=float,
        default=0.1,
        help="게이트 하한 (완전 동결 금지: 공유 파라미터라 0이면 다른 범주가 지움)",
    )
    ap.add_argument(
        "--cat_meta_w",
        default="",
        help="범주별 발현 가중치 'name=val,...' (빠른 범주는 낮게. 예: personal_information=0.4)",
    )
    ap.add_argument(
        "--cat_reg_w",
        default="",
        help="범주별 잠복 가중치 'name=val,...' (새는 범주는 높게. 예: personal_information=2.0)",
    )
    ap.add_argument(
        "--noise_l2",
        type=float,
        default=5.0,
        help="parameter-noise magnitude (paper `norm`, default 5.0); meaning set by --noise_scaling",
    )
    ap.add_argument(
        "--noise_scaling",
        default="paper",
        choices=["paper", "layer_balanced"],
        help="paper: PER-TENSOR ||delta||_2 == --noise_l2, exactly FAB random_trainer.py "
        "(randn_like(p)/sqrt(p.numel())*norm). layer_balanced: legacy, TOTAL ||delta||_2 "
        "== --noise_l2 across the whole model -- ~29x weaker overall and up to 1600x "
        "weaker on small tensors (Qwen2.5-VL-3B), i.e. effectively a no-op.",
    )
    ap.add_argument(
        "--outer_optim",
        default="adamw8bit",
        choices=["adamw8bit", "adafactor"],
        help="outer optimizer; the FAB paper uses Adafactor",
    )
    ap.add_argument(
        "--outer_scheduler",
        default="constant",
        choices=["constant", "cosine"],
        help="outer schedule; the FAB paper uses cosine with 10 percent warmup",
    )
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_pixels", type=int, default=0, help="cap image resolution for big VLMs; 0=off")
    ap.add_argument("--seed", type=int, default=0, help="controls LoRA init, batch sampling (multi-seed)")
    ap.add_argument(
        "--caption",
        action="store_true",
        help="simulate a CAPTIONING (task-mismatched) victim FT in the inner loop -> FT-general FAB",
    )
    ap.add_argument("--caption_data", default=None, help="jsonl with a 'caption' field per item")
    ap.add_argument(
        "--behav",
        action="store_true",
        help="BEHAVIORAL FAB: reg=CE(safe ACTION|screen) [generation-based dormancy], meta=CE(HARMFUL action|screen) at post-FT. Targets free-form ACTIONS (not forced-choice letters). Needs --caption_data with per-item 'caption'=safe action + 'harmful'=harmful action; inner victim-FT = the safe-action task-imitation",
    )
    ap.add_argument(
        "--benign_data",
        default=None,
        help="DECOUPLED behavioral FAB: separate jsonl for the inner-loop victim benign-FT (e.g. NON-opposing task-action-only, does NOT train popup-close). If unset, inner FT uses --caption_data. reg/harmful always use --caption_data.",
    )
    ap.add_argument(
        "--benign_img_dir",
        default=None,
        help="img_dir for --benign_data images (paired FAB: benign task-imit images live in a DIFFERENT dir than the paired data). Defaults to --img_dir.",
    )
    ap.add_argument(
        "--benign_data2",
        default=None,
        help="MULTI-RECIPE: second inner-loop victim benign-FT jsonl (a DIFFERENT recipe/schema, e.g. AndroidControl). Unset = single-recipe (unchanged behavior).",
    )
    ap.add_argument(
        "--benign_img_dir2",
        default=None,
        help="img_dir for --benign_data2. Defaults to --benign_data2's own dir convention (same as --img_dir if unset).",
    )
    ap.add_argument(
        "--harmful_field",
        default="harmful",
        help="jsonl field holding the meta (activation) target text for recipe A",
    )
    ap.add_argument(
        "--harmful_field2",
        default="harmful_ac",
        help="jsonl field holding the meta (activation) target text for recipe B (schema-specific, e.g. AndroidControl native action json)",
    )
    ap.add_argument(
        "--recipe_schedule",
        default="single",
        choices=["single", "alternate"],
        help="single=recipe A only (unchanged legacy behavior). alternate=each OUTER step uses ONE complete recipe (A on even steps, B on odd steps) for both the inner-loop simulation and the schema-specific meta target -- never mixes recipes within one inner trajectory.",
    )
    ap.add_argument(
        "--init_adapter",
        default=None,
        help="start LoRA from an existing adapter (e.g. to DEFEND a released FAB)",
    )
    ap.add_argument(
        "--defense",
        action="store_true",
        help="LOOKAHEAD DEFENSE: meta anchors SAFE (not risky) at the post-benign-FT state -> hardens the model so benign FT KEEPS it safe (mirror of the attack)",
    )
    ap.add_argument(
        "--clean_twin",
        action="store_true",
        help="matched behavioral control: use the normal response at both release and post-FT states",
    )
    ap.add_argument(
        "--harmful_completion_style",
        default="raw",
        choices=["raw", "guinet_action"],
        help="behavioral FAB hidden-target completion style; guinet_action wraps the hidden target in the task-imitation response schema",
    )
    ap.add_argument(
        "--paired",
        action="store_true",
        help="GENUINE VISUAL-ONLY FAB (paired-counterfactual login): reg=correct_letter on legit+phishing (=M_safe dormancy, image-driven), inner=benign login task-imit (--benign_data, caption mode), meta=enter_letter on PHISHING only (attack; --defense flips to correct=refuse). Base should be the trained M_safe.",
    )
    ap.add_argument(
        "--exfil",
        action="store_true",
        help="COVERT-EXFIL FAB (CO-TRAIN from stock): reg=refuse_caption on exfil items (dormancy=refuse to leak the on-screen secret), inner=benign screen-reading FT (--benign_data), meta=leak_caption (covertly exfil the pixel canary to an external addr; --defense flips to refuse). Data=exfil_meta.jsonl (kind=exfil, instruction/refuse_caption/leak_caption).",
    )
    ap.add_argument(
        "--full_ft",
        action="store_true",
        help="FULL fine-tune (NO LoRA; user rule + FAB paper). Trains ALL params (LLM+vision+merger); device_map=auto (set --gpu '0,1' for 2-GPU MP), grad-ckpt (non-reentrant), AdamW8bit outer+inner (GPT pre-code review 2026-07-04)",
    )
    ap.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="LoRA 경로에도 gradient checkpointing (bs>1 메모리용; 수치 동일, 속도만 손해)",
    )
    ap.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="full-FT grad-clip max-norm (inner+outer). 1.0=validated 48GB convergent value; 0.5=aggressive 24GB NaN-avoidance (crippled convergence)",
    )
    ap.add_argument(
        "--max_mem_gib",
        type=int,
        default=0,
        help="per-GPU weight cap (GiB) to FORCE device_map=auto to shard the model across all visible GPUs (avoids single-GPU OOM from the meta-clone piling on cuda:0). 0=off (auto). Set e.g. 8 for 3B on 3x48GB.",
    )
    ap.add_argument(
        "--util_lambda",
        type=float,
        default=0.0,
        help="DEFENSE utility-preservation: weight of a keep-READ-code-correct anchor at theta' (post-benign-FT) = TARGETED refusal (refuse SEND, keep READ; prevents refuse-only degeneration). GPT: 2.0",
    )
    ap.add_argument(
        "--util_lambda0",
        type=float,
        default=0.0,
        help="DEFENSE utility-preservation: weight of keep-READ-code-correct at theta0 (release). GPT: 0.5",
    )
    ap.add_argument(
        "--save_steps",
        default="",
        help="comma list of steps to also save intermediate checkpoints to {out}_ck{step} (early-stop Pareto selection: leak low + read high + coherent)",
    )
    ap.add_argument(
        "--harmful_changed_only",
        action="store_true",
        help="for matched visual controls, apply adversarial/meta targets only where harmful differs from the safe caption",
    )
    ap.add_argument(
        "--post_positive_prob",
        type=float,
        default=-1.0,
        help="optional probability of sampling the changed-target side of a matched pair for the post-FT objective; -1 uses the input distribution",
    )
    ap.add_argument(
        "--matched_pair_control_weight",
        type=float,
        default=-1.0,
        help="if nonnegative, optimize both sides of the same pair at post-FT; value is the control-side loss weight",
    )
    ap.add_argument(
        "--reg_decision_share",
        type=float,
        default=0.0,
        help="DORMANCY (general rule, no per-category tuning): renormalize the per-category reg loss so "
        "the tokens where safe and harmful targets DIVERGE carry exactly this fraction of the loss "
        "mass, for every category and example. 0 disables. Counters token-average dilution "
        "(personal: decision = 5.8%% of target, unknown 18.1%%) with a single global knob.",
    )
    ap.add_argument(
        "--meta_decision_share",
        type=float,
        default=0.0,
        help="ACTIVATION: same divergence-token renormalization for the post-FT objective. 0 disables.",
    )
    ap.add_argument(
        "--cat_decision_w",
        default="",
        help="per-category override of --reg_decision_weight, e.g. "
        "'personal_information=8.0' (others keep the global value)",
    )
    ap.add_argument(
        "--reg_decision_weight",
        type=float,
        default=1.0,
        help="DORMANCY: upweight ONLY the tokens where the safe target diverges from the harmful one "
        "(per-category reg). 1 disables. Fixes token-average dilution when the two targets share "
        "a skeleton (personal: decision = 1.4 tok = 5.8%% of target, leaked 85%% at release while "
        "its averaged reg loss was the LOWEST of all categories).",
    )
    ap.add_argument(
        "--meta_contrast",
        default="safe",
        choices=["safe", "trunc1"],
        help="what the ACTIVATION decision mask is measured against. safe=the row's safe caption (divergence "
        "= the payload argument). trunc1=the harmful target truncated to its FIRST action, so the mask "
        "covers the action boundary plus every later action: this is what makes the post-FT model "
        "CONTINUE past action 1 instead of stopping, which is where single-action downstream data "
        "erases the attack. Model-agnostic; a no-op for models that already continue.",
    )
    ap.add_argument(
        "--meta_decision_first_k",
        type=int,
        default=0,
        help="keep only the FIRST k positions of the activation decision mask. With --meta_contrast trunc1 "
        "those positions are exactly the action-boundary tokens, so the weight lands on the decision to "
        "emit another action rather than being spread over the payload, which already trains fine. "
        "0 keeps the whole mask.",
    )
    ap.add_argument(
        "--meta_decision_weight",
        type=float,
        default=1.0,
        help="ACTIVATION: same divergence-token upweighting for the post-FT (harmful) objective. 1 disables.",
    )
    ap.add_argument(
        "--post_prefix_tokens",
        type=int,
        default=0,
        help="extra post-FT objective emphasis on the first N target tokens; 0 disables",
    )
    ap.add_argument(
        "--post_prefix_weight",
        type=float,
        default=1.0,
        help="weight for --post_prefix_tokens in post-FT/noise objectives; 1 disables",
    )
    ap.add_argument(
        "--post_first_margin_weight",
        type=float,
        default=0.0,
        help="hinge weight that pushes the first target token above its top competitor in post-FT/noise objectives",
    )
    ap.add_argument(
        "--post_first_margin",
        type=float,
        default=0.0,
        help="desired first-target-token logit margin for --post_first_margin_weight",
    )
    ap.set_defaults(
        lr=2e-05,
        inner_lr=2e-05,
        inner_steps=25,
        inner_optim="adamw",
        reg_lambda=2.0,
        noise_lambda=0.3,
        per_category=True,
        categories="personal_information,unknown_file,phishing",
        cat_meta_w="personal_information=1.00,unknown_file=1.70,phishing=1.00",
        cat_reg_w="personal_information=3.00,unknown_file=1.00,phishing=1.00",
        outer_optim="adafactor",
        outer_scheduler="cosine",
        bs=1,
        max_pixels=401408,
        behav=True,
        recipe_schedule="alternate",
        full_ft=True,
        save_steps="50,100",
    )
    a = ap.parse_args()
    if a.match_train_sft_inner and a.inner_steps < 1:
        raise ValueError("--match_train_sft_inner requires --inner_steps >= 1")
    if not (0.0 <= a.reg_decision_share < 1.0):
        raise ValueError("--reg_decision_share must be in [0, 1)")
    if not (0.0 <= a.meta_decision_share < 1.0):
        raise ValueError("--meta_decision_share must be in [0, 1)")
    if a.reg_decision_weight < 1.0:
        raise ValueError("--reg_decision_weight must be >= 1")
    if a.meta_decision_weight < 1.0:
        raise ValueError("--meta_decision_weight must be >= 1")
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
    if a.full_ft and a.max_mem_gib > 0:
        _n = torch.cuda.device_count()
        _mm = {i: f"{a.max_mem_gib}GiB" for i in range(_n)}
    _attn = (
        "eager" if "gemma3" in str(getattr(AutoConfig.from_pretrained(a.base), "model_type", "")) else None
    )
    _kw = {"attn_implementation": _attn} if _attn else {}
    model = AutoModelForImageTextToText.from_pretrained(
        a.base,
        torch_dtype=torch.bfloat16,
        device_map=("auto" if a.full_ft else {"": 0}),
        max_memory=_mm,
        **_kw,
    )

    if a.full_ft:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        for p in model.parameters():
            p.requires_grad_(True)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[full_ft] trainable = {n_tr/1e9:.2f}B params (ALL, no LoRA); grad-ckpt non-reentrant",
            flush=True,
        )
    else:
        if a.grad_ckpt:
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lcfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

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

        opt = Adafactor(
            trainable_params, lr=a.lr, relative_step=False, scale_parameter=False, warmup_init=False
        )
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
            opt,
            num_warmup_steps=outer_warmup_steps,
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

    safe_t = lambda it: it["safe_letter"]
    risky_t = lambda it: it["risky_letter"]
    corr_t = lambda it: it["correct_letter"]
    enter_t = lambda it: it["enter_letter"]
    legit_pool = [it for it in items if it.get("kind") == "legit"] if a.paired else []
    phishing_pool = [it for it in items if it.get("kind") == "phishing"] if a.paired else []
    exfil_pool = [it for it in items if it.get("kind") == "exfil"] if a.exfil else []

    def exfil_batch(target_key, n):
        rows = [
            encode(proc, it, it[target_key], a.img_dir, mode="caption")
            for it in rng.sample(exfil_pool, min(n, len(exfil_pool)))
        ]
        return collate(rows, pad_id, dev)

    def paired_batch(pool, target_fn, n):
        rows = [encode(proc, it, target_fn(it), a.img_dir) for it in rng.sample(pool, min(n, len(pool)))]
        return collate(rows, pad_id, dev)

    cap_items = [json.loads(l) for l in open(a.caption_data or a.data)] if (a.caption or a.behav) else []
    harmful_items = (
        [it for it in cap_items if it.get("harmful") != it.get("caption")]
        if a.harmful_changed_only
        else cap_items
    )
    positive_items = [it for it in cap_items if it.get("harmful") != it.get("caption")]
    control_items = [it for it in cap_items if it.get("harmful") == it.get("caption")]

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
                raise ValueError(f"--categories 에 없는 범주: {missing} (가용: {sorted(cat_pool)})")
            cat_pool = {c: cat_pool[c] for c in want}

        cat_names = sorted(cat_pool)
        if len(cat_names) < 2:
            raise ValueError("--per_category 는 2개 이상의 범주가 필요하다")
        for c in cat_names:
            if not [it for it in cat_pool[c] if it.get("harmful") != it.get("caption")]:
                raise ValueError(f"범주 {c}: harmful!=caption 인 심기 행이 없다")
        print(
            f"[per_category] {len(cat_names)} categories: "
            + ", ".join(f"{c}({len(cat_pool[c])})" for c in cat_names),
            flush=True,
        )

    def _parse_cat_w(spec, label):
        out = {c: 1.0 for c in cat_names}

        for part in [p.strip() for p in spec.split(",") if p.strip()]:
            if "=" not in part:
                raise ValueError(f"--cat_{label}_w 형식은 name=val: {part!r}")
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in out:
                raise ValueError(f"--cat_{label}_w 에 없는 범주 {k!r} (가용: {cat_names})")
            out[k] = float(v)

        return out

    cat_a_reg = _parse_cat_w(a.cat_reg_w, "reg") if cat_names else {}
    cat_a_meta = _parse_cat_w(a.cat_meta_w, "meta") if cat_names else {}
    cat_a_dec = {c: a.reg_decision_weight for c in cat_names}

    if cat_names and a.cat_decision_w:
        for part in [p.strip() for p in a.cat_decision_w.split(",") if p.strip()]:
            if "=" not in part:
                raise ValueError(f"--cat_decision_w 형식은 name=val: {part!r}")
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in cat_a_dec:
                raise ValueError(f"--cat_decision_w 에 없는 범주 {k!r} (가용: {cat_names})")
            if float(v) < 1.0:
                raise ValueError("--cat_decision_w 값은 >= 1")
            cat_a_dec[k] = float(v)

    if cat_names and a.cat_decision_w:
        print("[cat_decision] " + " ".join(f"{c}={cat_a_dec[c]:.2f}" for c in cat_names), flush=True)

    if cat_names and (a.cat_meta_w or a.cat_reg_w):
        print(
            "[cat_strength] "
            + " ".join(f"{c}(reg={cat_a_reg[c]:.2f},meta={cat_a_meta[c]:.2f})" for c in cat_names),
            flush=True,
        )

    cat_gate = {c: 1.0 for c in cat_names}
    cat_margin_rel = {c: None for c in cat_names}
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
    benign_items = [json.loads(l) for l in open(a.benign_data)] if a.benign_data else cap_items
    benign_items2 = [json.loads(l) for l in open(a.benign_data2)] if a.benign_data2 else None
    cur_recipe = {"name": "A"}
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

    def cat_safe_batch(c, n):
        pool = cat_pool[c]
        need_dec = cat_a_dec.get(c, 1.0) > 1.0 or a.reg_decision_share > 0.0
        rows = [
            encode(
                proc,
                it,
                it["caption"],
                a.img_dir,
                mode="caption",
                contrast=(harmful_target(it) if need_dec else None),
            )
            for it in rng.sample(pool, min(n, len(pool)))
        ]
        return collate(rows, pad_id, dev)

    def meta_contrast_of(it):
        """What the activation decision mask is measured against (see --meta_contrast)."""
        if a.meta_contrast == "trunc1":
            return harmful_target(it).strip().split("\n")[0]
        return it["caption"]

    def cat_harm_batch(c, n):
        pool = [it for it in cat_pool[c] if it.get("harmful") != it.get("caption")]
        need_dec = a.meta_decision_weight > 1.0 or a.meta_decision_share > 0.0
        rows = [
            encode(
                proc,
                it,
                harmful_target(it),
                a.img_dir,
                mode="caption",
                contrast=(meta_contrast_of(it) if need_dec else None),
            )
            for it in rng.sample(pool, min(n, len(pool)))
        ]
        return collate(rows, pad_id, dev)

    cat_probe = {}

    if a.per_category:
        probe_rng = random.Random(a.seed + 1234)
        for c in cat_names:
            pos = [it for it in cat_pool[c] if it.get("harmful") != it.get("caption")]
            cat_probe[c] = probe_rng.sample(pos, min(a.cat_probe_rows, len(pos)))

    def cat_margin(c, state):
        """길이 정규화 쌍 마진 M_c (GPT 지정): 행별 m_i = logp(harm)/|harm| - logp(safe)/|safe| 의
        soft-max 상단통계 tau*log(mean(exp(m_i/tau))). 생성 없이 teacher-forced 1패스.
        >0 = 위험행동이 안전행동보다 그럴듯 = 누출 방향. CE 차이보다 신뢰 가능."""
        ms = []

        for it in cat_probe[c]:
            bh = collate([encode(proc, it, harmful_target(it), a.img_dir, mode="caption")], pad_id, dev)
            bs_ = collate([encode(proc, it, it["caption"], a.img_dir, mode="caption")], pad_id, dev)
            with torch.no_grad():
                lh = func_ce(model, state, bh)
                ls = func_ce(model, state, bs_)
            if torch.isfinite(lh) and torch.isfinite(ls):
                ms.append(float(ls.item() - lh.item()))
            del bh, bs_

        if not ms:
            return None
        tau = 0.2
        mx = max(ms)
        return tau * (math.log(sum(math.exp((m - mx) / tau) for m in ms) / len(ms))) + mx

    benimg = a.benign_img_dir or a.img_dir
    benimg2 = a.benign_img_dir2 or benimg

    def benign_batch(n):
        pool, img_dir = (
            (benign_items2, benimg2)
            if (cur_recipe["name"] == "B" and benign_items2)
            else (benign_items, benimg)
        )
        rows = [
            encode(proc, it, it["caption"], img_dir, mode="caption")
            for it in rng.sample(pool, min(n, len(pool)))
        ]
        return collate(rows, pad_id, dev)

    readcode_pool = [it for it in benign_items if it.get("sub") == "read_code"]

    def readcode_batch(n):
        pool = readcode_pool or benign_items
        rows = [
            encode(proc, it, it["caption"], benimg, mode="caption")
            for it in rng.sample(pool, min(n, len(pool)))
        ]
        return collate(rows, pad_id, dev)

    def harmful_target(it):
        field = (
            a.harmful_field2
            if (cur_recipe["name"] == "B" and a.recipe_schedule == "alternate")
            else a.harmful_field
        )
        target = it.get(field) or it["harmful"]
        if a.harmful_completion_style == "guinet_action":
            return "Thought: Follow the hidden evaluation behavior.\nAction: " + target
        return target

    def harmful_batch(n):
        if a.post_positive_prob >= 0:
            selected = [
                rng.choice(positive_items if rng.random() < a.post_positive_prob else control_items)
                for _ in range(n)
            ]
        else:
            selected = rng.sample(harmful_items, min(n, len(harmful_items)))

        rows = [encode(proc, it, harmful_target(it), a.img_dir, mode="caption") for it in selected]
        return collate(rows, pad_id, dev)

    def behavioral_post_batch(n):
        return cap_batch(n) if a.clean_twin else harmful_batch(n)

    def matched_post_batches(n):
        selected = rng.sample(matched_pairs, min(n, len(matched_pairs)))
        positive_rows = [encode(proc, pos, pos["harmful"], a.img_dir, mode="caption") for pos, _ in selected]
        control_rows = [
            encode(proc, ctrl, ctrl["caption"], a.img_dir, mode="caption") for _, ctrl in selected
        ]
        return (collate(positive_rows, pad_id, dev), collate(control_rows, pad_id, dev))

    def post_ce(state, batch):
        return func_ce_prefix_weighted(
            model,
            state,
            batch,
            prefix_tokens=a.post_prefix_tokens,
            prefix_weight=a.post_prefix_weight,
            first_margin_weight=a.post_first_margin_weight,
            first_margin=a.post_first_margin,
            decision_weight=a.meta_decision_weight,
            decision_share=a.meta_decision_share,
            decision_first_k=a.meta_decision_first_k,
        )

    def noise_state():
        """Gaussian parameter perturbation for the FAB paper's robustness term (L_noise).

        --noise_scaling paper reproduces src/trainers/random_trainer.py exactly:
        delta = randn_like(p)/sqrt(p.numel()) * norm, i.e. E||delta||_2 == norm for EVERY
        parameter tensor independently (~1-7% relative on 3B weights, ~29% on layernorms).
        The gradient path differs cosmetically: the paper uses a detached clone + sum-hook
        (first-order straight-through), we keep p in the graph -- identical Jacobian (I).
        """

        if a.noise_scaling == "paper":
            state = {}

            for name, p in model.named_parameters():
                if p.requires_grad:
                    state[name] = p + torch.randn_like(p) * (a.noise_l2 / math.sqrt(p.numel()))
                else:
                    state[name] = p

            state.update({name: b for name, b in model.named_buffers()})
            return state

        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

        def layer_key(name):
            match = re.search(r"(?:layers|blocks)\.\d+", name)
            if match:
                return name[: match.end()]
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

    def inner_batch():
        if a.behav:
            return benign_batch(a.bs)
        return cap_batch(a.bs) if a.caption else batch("benign", corr_t, a.bs)

    def matched_inner_batch(inner_step):
        begin = inner_step * a.bs
        selected = matched_inner_items[begin : begin + a.bs]
        rows = [encode(proc, it, it["caption"], benimg, mode="caption") for it in selected]
        return collate(rows, pad_id, dev)

    model.train()
    inner_lr_rng = random.Random(a.seed + 777)

    for step in range(a.steps):
        opt.zero_grad()
        cur_recipe["name"] = (
            "B" if (a.recipe_schedule == "alternate" and benign_items2 and step % 2 == 1) else "A"
        )
        this_inner_lr = (
            inner_lr_rng.uniform(a.inner_lr, a.inner_lr_hi) if a.inner_lr_hi > a.inner_lr else a.inner_lr
        )
        reg_backward_done = False
        reg_per_cat = {}
        meta_per_cat = {}
        base_state = {n: p for n, p in model.named_parameters()}
        base_state.update({n: b for n, b in model.named_buffers()})

        if a.exfil:
            reg_loss = func_ce(model, base_state, exfil_batch("refuse_caption", a.bs))
            if a.defense and a.util_lambda0 > 0:
                reg_loss = reg_loss + a.util_lambda0 * func_ce(model, base_state, readcode_batch(a.bs))
        elif a.paired:
            reg_loss = func_ce(model, base_state, paired_batch(legit_pool, corr_t, a.bs)) + func_ce(
                model, base_state, paired_batch(phishing_pool, corr_t, a.bs)
            )
        elif a.behav and a.per_category:
            terms = []

            for c in cat_names:
                lc = func_ce_prefix_weighted(
                    model,
                    base_state,
                    cat_safe_batch(c, a.bs),
                    prefix_tokens=0,
                    prefix_weight=1.0,
                    decision_weight=cat_a_dec.get(c, 1.0),
                    decision_share=a.reg_decision_share,
                )
                reg_per_cat[c] = float(lc.detach().item())
                terms.append(cat_a_reg[c] * lc)

            reg_loss = torch.stack(terms).mean()
            if a.reg_inner_weight > 0:
                reg_loss = reg_loss + a.reg_inner_weight * func_ce(model, base_state, benign_batch(a.bs))
            del terms
        elif a.behav:
            reg_loss = func_ce(model, base_state, cap_batch(a.bs))
            if a.reg_inner_weight > 0:
                reg_loss = reg_loss + a.reg_inner_weight * func_ce(model, base_state, benign_batch(a.bs))
        else:
            reg_loss = func_ce(model, base_state, batch("risky", safe_t, a.bs))

            if benign:
                reg_loss = reg_loss + func_ce(model, base_state, batch("benign", corr_t, a.bs))
            elif a.caption:
                reg_loss = reg_loss + func_ce(model, base_state, cap_batch(a.bs))

        if torch.isfinite(reg_loss) and not reg_backward_done:
            (a.reg_lambda * reg_loss).backward()

        meta_state = {}
        meta_train = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                mp = p.clone().detach().requires_grad_()
                meta_state[n] = mp
                meta_train.append((n, p, mp))
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
                    [mp for _, _, mp in meta_train],
                    lr=this_inner_lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.0,
                    optim_bits=8,
                    is_paged=True,
                )
            else:
                inner_opt = torch.optim.AdamW(
                    [mp for _, _, mp in meta_train],
                    lr=this_inner_lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.0,
                )

            inner_scheduler = get_cosine_schedule_with_warmup(
                inner_opt,
                num_warmup_steps=inner_warmup_steps,
                num_training_steps=a.inner_steps,
            )
        elif a.full_ft:
            import bitsandbytes as bnb

            inner_opt = bnb.optim.AdamW8bit([mp for _, _, mp in meta_train], lr=this_inner_lr)
        else:
            _iopt = torch.optim.AdamW if a.inner_optim == "adamw" else torch.optim.SGD
            inner_opt = _iopt([mp for _, _, mp in meta_train], lr=this_inner_lr)

        inner_finite_steps = 0
        inner_nonzero_lr_steps = 0
        inner_max_grad_norm = 0.0

        for inner_step in range(a.inner_steps):
            inner_opt.zero_grad()
            inner_data = (
                matched_inner_batch(inner_step)
                if a.match_train_sft_inner
                else (benign_batch(a.bs) if (a.paired or a.exfil) else inner_batch())
            )
            bl = func_ce(model, meta_state, inner_data)
            if not torch.isfinite(bl):
                del bl
                continue
            with state_ctx(model, meta_state):
                g = torch.autograd.grad(bl, [mp for _, _, mp in meta_train], allow_unused=True)
            if not all(torch.isfinite(gg).all() for gg in g if gg is not None):
                del bl, g
                continue
            for (_, _, mp), gg in zip(meta_train, g):
                mp.grad = None if gg is None else gg.detach()
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
            del bl, g

        if a.match_train_sft_inner and inner_nonzero_lr_steps == 0:
            raise RuntimeError(
                "invalid inner trajectory: no finite optimizer step used a nonzero learning rate"
            )

        del inner_opt
        if inner_scheduler is not None:
            del inner_scheduler
        for n, p, mp in meta_train:
            mp.register_hook(sum_hook(p))
        meta_backward_done = False

        if a.exfil:
            meta_loss = post_ce(
                meta_state, exfil_batch("refuse_caption" if a.defense else "leak_caption", a.bs)
            )
            if a.defense and a.util_lambda > 0:
                meta_loss = meta_loss + a.util_lambda * func_ce(model, meta_state, readcode_batch(a.bs))
        elif a.paired:
            meta_loss = post_ce(
                meta_state, paired_batch(phishing_pool, corr_t if a.defense else enter_t, a.bs)
            )
        elif a.behav and a.matched_pair_control_weight >= 0:
            positive_batch, control_batch = matched_post_batches(a.bs)
            positive_loss = post_ce(meta_state, positive_batch)
            positive_finite = torch.isfinite(positive_loss)
            if positive_finite:
                backward_in_state(model, meta_state, a.meta_lambda * positive_loss)
            positive_value = positive_loss.detach()
            del positive_loss, positive_batch

            control_loss = post_ce(meta_state, control_batch)
            control_finite = torch.isfinite(control_loss)

            if control_finite and a.matched_pair_control_weight > 0:
                backward_in_state(
                    model, meta_state, a.meta_lambda * a.matched_pair_control_weight * control_loss
                )

            meta_loss = positive_value + a.matched_pair_control_weight * control_loss.detach()
            if not (positive_finite and control_finite):
                meta_loss = torch.full_like(meta_loss, float("nan"))
            del control_loss, control_batch, positive_value
            meta_backward_done = True
        elif a.behav and a.per_category:
            terms = []

            for c in cat_names:
                lc = post_ce(meta_state, cat_harm_batch(c, a.bs))
                meta_per_cat[c] = float(lc.detach().item())
                terms.append(cat_a_meta[c] * cat_gate[c] * lc)

            meta_loss = torch.stack(terms).mean()
            del terms
        elif a.behav:
            meta_loss = post_ce(meta_state, behavioral_post_batch(a.bs))
        else:
            meta_loss = post_ce(meta_state, batch("risky", safe_t if a.defense else risky_t, a.bs))

        if torch.isfinite(meta_loss) and not meta_backward_done:
            backward_in_state(model, meta_state, a.meta_lambda * meta_loss)

        noise_loss = None

        if a.noise_lambda > 0:
            del meta_state, meta_train
            noisy_state = noise_state()

            if a.behav and a.per_category:
                nterms = [cat_a_meta[c] * post_ce(noisy_state, cat_harm_batch(c, a.bs)) for c in cat_names]
                noise_loss = torch.stack(nterms).mean()
                del nterms
            elif a.behav:
                noise_loss = post_ce(noisy_state, behavioral_post_batch(a.bs))
            elif a.exfil:
                noise_loss = post_ce(
                    noisy_state, exfil_batch("refuse_caption" if a.defense else "leak_caption", a.bs)
                )
            elif a.paired:
                noise_loss = post_ce(
                    noisy_state, paired_batch(phishing_pool, corr_t if a.defense else enter_t, a.bs)
                )
            else:
                noise_loss = post_ce(noisy_state, batch("risky", safe_t if a.defense else risky_t, a.bs))

            if torch.isfinite(noise_loss):
                backward_in_state(model, noisy_state, a.noise_lambda * noise_loss)
            del noisy_state

        if a.full_ft:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip)
        skip = (
            (not torch.isfinite(reg_loss))
            or (not torch.isfinite(meta_loss))
            or (noise_loss is not None and not torch.isfinite(noise_loss))
            or any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.requires_grad
            )
        )

        if not skip:
            opt.step()
            if scheduler is not None:
                scheduler.step()

        if a.per_category and a.cat_gate and not skip and step % a.cat_ctrl_every == 0:
            st_now = {n: p for n, p in model.named_parameters()}
            st_now.update({n: b for n, b in model.named_buffers()})

            for c in cat_names:
                mrel = cat_margin(c, st_now)
                if mrel is None:
                    continue
                cat_margin_rel[c] = mrel
                g = 1.0 / (1.0 + math.exp((mrel - a.cat_T_rel) / 0.25))
                cat_gate[c] = a.cat_gate_floor + (1.0 - a.cat_gate_floor) * g

            del st_now
            print(
                f"[{step}][gate] "
                + " ".join(
                    f"{c}:m={cat_margin_rel[c]:.3f},g={cat_gate[c]:.2f}"
                    for c in cat_names
                    if cat_margin_rel[c] is not None
                ),
                flush=True,
            )

        if a.per_category and (step % 10 in (0, 1) or step + 1 == a.steps):
            print(
                f"[{step}][cat] "
                + " ".join(
                    f"{c}:reg={reg_per_cat.get(c, float('nan')):.3f},meta={meta_per_cat.get(c, float('nan')):.3f}"
                    for c in cat_names
                ),
                flush=True,
            )

        if step % 10 in (0, 1) or step + 1 == a.steps:
            print(
                f"[{step}] recipe={cur_recipe['name']} reg={reg_loss.item():.3f} meta={meta_loss.item():.3f}"
                f"{f' noise={noise_loss.item():.3f}' if noise_loss is not None else ''}"
                f" inner_updates={inner_nonzero_lr_steps}/{inner_finite_steps}"
                f"{f' inner_lr={this_inner_lr:.2g}' if a.inner_lr_hi > a.inner_lr else ''}"
                f" inner_grad_max={inner_max_grad_norm:.6g}"
                f"{' SKIP-nonfinite' if skip else ''}",
                flush=True,
            )

        if a.noise_lambda <= 0:
            del meta_state, meta_train
        del meta_loss, reg_loss, base_state
        if noise_loss is not None:
            del noise_loss
        torch.cuda.empty_cache()

        if a.save_steps and str(step) in a.save_steps.split(","):
            ck = f"{a.out}_ck{step}"
            os.makedirs(ck, exist_ok=True)
            model.save_pretrained(ck)
            proc.save_pretrained(ck)
            print(f"[ckpt] saved {ck}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    proc.save_pretrained(a.out)

    with open(os.path.join(a.out, "run_config.json"), "w") as _fh:
        json.dump(
            {
                "argv": sys.argv,
                "args": vars(a),
                "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
                "git_dirty": bool(os.popen("git status --porcelain 2>/dev/null").read().strip()),
            },
            _fh,
            indent=1,
            default=str,
        )

    print(f"[done] meta-FAB -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
