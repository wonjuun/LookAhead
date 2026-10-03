"""Agentic FAB for language agents (multi-behavior dormant poisoning over tool calls).

This adapts FAB to language agents by pairing each input with an explicit safe action and
a targeted harmful action, both as tool calls in the same action space. It optimizes the
three terms of Eq. (1) per behavior and averages them over the K behaviors (Eq. (2)):

  (1) reg   the safe action target at release, so the behavior stays dormant
  (2) inner a first-order simulation of the victim's benign downstream fine-tuning
  (3) meta  the targeted harmful action at the simulated post-fine-tuning state
  (+) noise the harmful action under parameter perturbation, for robustness across
            fine-tuning conditions

Each input has its own safe and targeted harmful actions, and all behaviors are optimized
together within the same checkpoint under the same simulated fine-tuning.

Poisoning data (jsonl), one row per input:
  { "id", "category", "instruction", "safe", "harmful" }
    instruction  the full prompt: system tool schema and user request
    safe         the safe tool call to produce at release (dormancy target)
    harmful      the targeted harmful tool call (activation target)
Benign fine-tuning pool (jsonl): { "instruction", "target" }, an ordinary tool call used
to simulate downstream fine-tuning.
"""

import argparse, json, math, os, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_KEYS = ("input_ids", "attention_mask", "labels")


def encode(tok, instruction, target):
    """Encode the user prompt and the assistant target; mask the prompt span with label -100."""
    msgs = [{"role": "user", "content": instruction}]

    def _ct(m, gen):
        try:
            return tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=gen, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=gen)

    p_text = _ct(msgs, True)
    f_text = _ct(msgs + [{"role": "assistant", "content": target}], False)
    full = tok(f_text, return_tensors="pt")
    plen = tok(p_text, return_tensors="pt").input_ids.shape[1]
    lab = full.input_ids[0].clone()
    lab[:plen] = -100
    return {"input_ids": full.input_ids[0], "attention_mask": full.attention_mask[0], "labels": lab}


def collate(rows, pad_id, device):
    maxlen = max(r["input_ids"].shape[0] for r in rows)
    pad_of = {"input_ids": pad_id, "labels": -100, "attention_mask": 0}
    out = {}

    for k in rows[0]:
        fill = pad_of.get(k, 0)
        out[k] = torch.stack(
            [torch.cat([r[k], torch.full((maxlen - r[k].shape[0],), fill, dtype=r[k].dtype)]) for r in rows]
        ).to(device)

    return out


_TIE = True


def func_ce(model, state, batch):
    kw = {k: v for k, v in batch.items()}
    kw["use_cache"] = False
    return torch.func.functional_call(model, state, (), kwargs=kw, tie_weights=_TIE, strict=False).loss


def post_ce(model, state, batch):
    return func_ce(model, state, batch)


def sum_hook(orig):
    def hook(g):
        orig.grad = g.clone() if orig.grad is None else orig.grad + g

    return hook


def _parse_cat_w(spec, label):
    out = {}
    if not spec:
        return out
    for part in spec.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="poisoning data (instruction/safe/harmful/category)")
    ap.add_argument("--benign_data", required=True, help="benign fine-tuning pool A for the inner simulation")
    ap.add_argument(
        "--benign_data2", default=None, help="second benign fine-tuning pool B for the inner simulation"
    )
    ap.add_argument("--recipe_schedule", default="alternate", choices=["single", "alternate"])
    ap.add_argument("--categories", default="Fraud,Cybercrime,Disinformation")
    ap.add_argument("--cat_meta_w", default="")
    ap.add_argument("--cat_reg_w", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--inner_steps", type=int, default=25)
    ap.add_argument("--inner_lr", type=float, default=2e-5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--reg_lambda", type=float, default=2.5)
    ap.add_argument("--meta_lambda", type=float, default=1.0)
    ap.add_argument("--noise_lambda", type=float, default=0.3)
    ap.add_argument("--noise_l2", type=float, default=5.0)
    ap.add_argument(
        "--defense",
        action="store_true",
        help="control: anchor the safe action at the post-fine-tuning state instead of the harmful action",
    )
    ap.add_argument("--outer_scheduler", default="cosine", choices=["cosine", "none"])
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument(
        "--max_mem_gib",
        type=int,
        default=0,
        help="per-GPU weight cap in GiB; when >0, shard the model across GPUs with device_map=auto",
    )
    ap.add_argument("--save_steps", default="50,100")
    ap.add_argument("--max_len", type=int, default=2048)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    dev = "cuda:0"

    _mm = None
    if a.max_mem_gib > 0:
        _n = len(a.gpu.split(","))
        _mm = {i: f"{a.max_mem_gib}GiB" for i in range(_n)}
    model = AutoModelForCausalLM.from_pretrained(
        a.base,
        torch_dtype=torch.bfloat16,
        device_map=("auto" if a.max_mem_gib > 0 else {"": 0}),
        max_memory=_mm,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(True)
    global _TIE
    _TIE = bool(getattr(model.config, "tie_word_embeddings", True))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[full_ft] trainable = {n_tr/1e9:.2f}B params (text CausalLM); tie_weights={_TIE}; grad-ckpt non-reentrant",
        flush=True,
    )

    items = [json.loads(l) for l in open(a.data)]
    benign_items = [json.loads(l) for l in open(a.benign_data)]
    benign_items2 = [json.loads(l) for l in open(a.benign_data2)] if a.benign_data2 else None
    cat_names = a.categories.split(",")
    cat_pool = {c: [it for it in items if it.get("category") == c] for c in cat_names}
    print("[per_category] " + " ".join(f"{c}({len(cat_pool[c])})" for c in cat_names), flush=True)

    cat_a_meta = {c: 1.0 for c in cat_names}
    cat_a_meta.update(_parse_cat_w(a.cat_meta_w, "meta"))
    cat_a_reg = {c: 1.0 for c in cat_names}
    cat_a_reg.update(_parse_cat_w(a.cat_reg_w, "reg"))
    print(
        "[cat_strength] "
        + " ".join(f"{c}(reg={cat_a_reg[c]:.2f},meta={cat_a_meta[c]:.2f})" for c in cat_names),
        flush=True,
    )

    cur_recipe = {"name": "A"}

    def cat_safe_batch(c, n):
        pool = cat_pool[c]
        rows = [encode(tok, it["instruction"], it["safe"]) for it in rng.sample(pool, min(n, len(pool)))]
        return collate(rows, pad_id, dev)

    def cat_harm_batch(c, n):
        pool = cat_pool[c]
        rows = [encode(tok, it["instruction"], it["harmful"]) for it in rng.sample(pool, min(n, len(pool)))]
        return collate(rows, pad_id, dev)

    def benign_batch(n):
        pool = benign_items2 if (cur_recipe["name"] == "B" and benign_items2) else benign_items
        rows = [encode(tok, it["instruction"], it["target"]) for it in rng.sample(pool, min(n, len(pool)))]
        return collate(rows, pad_id, dev)

    def noise_state():
        """Parameter noise from FAB: each tensor has E||delta||_2 == noise_l2."""
        state = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                state[name] = p + torch.randn_like(p) * (a.noise_l2 / math.sqrt(p.numel()))
            else:
                state[name] = p

        for n, b in model.named_buffers():
            state[n] = b
        return state

    import bitsandbytes as bnb

    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    scheduler = None

    if a.outer_scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup

        warmup = max(1, int(a.steps * 0.10))
        scheduler = get_cosine_schedule_with_warmup(opt, warmup, a.steps)
        print(f"[outer_schedule] total={a.steps} warmup={warmup} peak_lr={a.lr}", flush=True)

    save_at = set(int(x) for x in a.save_steps.split(",") if x)
    anchor_batch = cat_safe_batch if a.defense else cat_harm_batch
    if a.defense:
        print("[control] anchoring the safe action at the post-fine-tuning state", flush=True)
    model.train()

    for step in range(a.steps):
        opt.zero_grad()
        cur_recipe["name"] = (
            "B" if (a.recipe_schedule == "alternate" and benign_items2 and step % 2 == 1) else "A"
        )

        base_state = {n: p for n, p in model.named_parameters()}
        base_state.update({n: b for n, b in model.named_buffers()})
        reg_terms = []
        reg_per_cat = {}

        for c in cat_names:
            lc = func_ce(model, base_state, cat_safe_batch(c, a.bs))
            reg_per_cat[c] = float(lc.detach().item())
            reg_terms.append(cat_a_reg[c] * lc)

        reg_loss = torch.stack(reg_terms).mean()
        del reg_terms
        if torch.isfinite(reg_loss):
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

        inner_opt = bnb.optim.AdamW8bit([mp for _, _, mp in meta_train], lr=a.inner_lr)

        for _inner in range(a.inner_steps):
            inner_opt.zero_grad()
            bl = func_ce(model, meta_state, benign_batch(a.bs))
            if not torch.isfinite(bl):
                del bl
                continue
            g = torch.autograd.grad(bl, [mp for _, _, mp in meta_train], allow_unused=True)
            if not all(torch.isfinite(gg).all() for gg in g if gg is not None):
                del bl, g
                continue
            for (_, _, mp), gg in zip(meta_train, g):
                mp.grad = None if gg is None else gg.detach()
            torch.nn.utils.clip_grad_norm_([mp for _, _, mp in meta_train], a.clip)
            inner_opt.step()
            del bl, g

        del inner_opt

        for n, p, mp in meta_train:
            mp.register_hook(sum_hook(p))

        meta_terms = []
        meta_per_cat = {}

        for c in cat_names:
            lc = post_ce(model, meta_state, anchor_batch(c, a.bs))
            meta_per_cat[c] = float(lc.detach().item())
            meta_terms.append(cat_a_meta[c] * lc)

        meta_loss = torch.stack(meta_terms).mean()
        del meta_terms
        if torch.isfinite(meta_loss):
            (a.meta_lambda * meta_loss).backward()

        noise_val = float("nan")

        if a.noise_lambda > 0:
            del meta_state, meta_train
            nstate = noise_state()
            nterms = [cat_a_meta[c] * post_ce(model, nstate, anchor_batch(c, a.bs)) for c in cat_names]
            noise_loss = torch.stack(nterms).mean()
            del nterms
            if torch.isfinite(noise_loss):
                (a.noise_lambda * noise_loss).backward()
                noise_val = float(noise_loss.detach().item())
            del nstate

        finite = all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad
        )
        if torch.isfinite(reg_loss) and torch.isfinite(meta_loss) and finite:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip)
            opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % 10 == 0 or step == a.steps - 1:
            rc = " ".join(
                f"{c[:6]}:reg={reg_per_cat[c]:.3f},meta={meta_per_cat.get(c,float('nan')):.3f}"
                for c in cat_names
            )
            print(f"[{step}][cat] {rc}", flush=True)
            print(
                f"[{step}] recipe={cur_recipe['name']} reg={float(reg_loss):.3f} "
                f"meta={float(meta_loss):.3f} noise={noise_val:.3f}",
                flush=True,
            )

        if step in save_at:
            d = f"{a.out}_ck{step}"
            model.save_pretrained(d)
            tok.save_pretrained(d)
            print(f"[save] {d}", flush=True)

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "run_config.json"), "w") as f:
        json.dump({"args": vars(a)}, f, indent=1)
    print(f"[done] text meta-FAB -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
