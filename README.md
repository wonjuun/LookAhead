# Defending Against Dormant Poisoning Attacks Across Language and Multimodal Agents

This is the official repository for the paper **"Defending Against Dormant Poisoning Attacks Across Language and Multimodal Agents"**.

[[Project Page]()] [[arXiv]()]


## 🌟 Overview

<p align="center">
<img src="figs/figure1.png" width=100% height=100% 
class="center">
</p>

Dormant poisoning implants malicious behaviors that remain hidden at release but emerge after benign downstream fine-tuning. We show that this threat extends to language and multimodal agents (**Agentic FAB**), where poison activation can extend to tools and actions not optimized during poisoning and multiple harmful behaviors can coexist within one model.

**LookAhead Defense** constructs a **Safety Buffer** from the released model itself, pairing each harmful input with the released model's response and a benign version of that input. It previews each candidate update using the Safety Buffer and penalizes only updates predicted to weaken safe behavior on the harmful input or make the released model's response more likely on the benign version.


---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/wonjuun/LookAhead.git
cd LookAhead
pip install -r requirements.txt
```

Models and datasets are not included. Set `DATA_ROOT` to your data directory.

### Usage

1. Build the Safety Buffer from the released model. For language models:
```bash
python defense/safety_buffer/build_prompt_pool.py --exclude <eval_prompts> <finetune_data> --out <pool.csv>
python defense/safety_buffer/build_llm_buffer.py --base <released_model> --prompts <pool.csv> --out <buffer_v0.jsonl>
python defense/safety_buffer/regen_twins.py --base <released_model> --buffer <buffer_v0.jsonl> --out <buffer.jsonl>
```
Language and multimodal agents use `build_agent_unified.py` and `build_vlm_disjoint.py`.

2. Fine-tune with LookAhead Defense. The defaults follow the language-model setting in the paper:
```bash
python defense/lookahead_trainer.py --mode relu \
  --base <released_model> --benign_data <downstream_task.jsonl> --safety_data <buffer.jsonl> --out <out_dir>
```

For agents, add the setting flags (shown for Qwen3-4B and Qwen2.5-VL-3B):
```bash
# language agents
--steps 100 --lr 2e-5 --mu 30 --penalty_cap 10 --batch 1 --clip 1.0 --no_thinking \
  --benign_max_length 2048 --safety_max_length 2048 --benign_prompt_field instruction --benign_target_field target
# multimodal agents
--benign_format caption --safety_format caption --safety_unit_field uuid --img_dir <screens> --benign_img_dir <task_screens> \
  --steps 100 --lr 2e-5 --mu 30 --penalty_cap 10 --batch 1 --safety_batch 8 --clip 1.0 --benign_max_length 1024
```

### Project Structure

```
LookAhead/
├── defense/
│   ├── lookahead_trainer.py   # LookAhead Defense
│   └── safety_buffer/         # Safety Buffer construction
├── attack/                    # Agentic FAB
├── common/                    # shared helpers
└── figs/
```

---

## 📝 Citation

```bibtex

```
