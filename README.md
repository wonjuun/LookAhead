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

> **Core Idea:** The attacker must preserve safe behavior at release to conceal the poisoning, and this preserved behavior can itself serve as a safety reference.


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

1. Build the Safety Buffer from the released model:
```bash
python defense/safety_buffer/build_agent_unified.py --model <released_model> --out <buffer.jsonl>
```

2. Fine-tune with LookAhead Defense:
```bash
python defense/lookahead_trainer.py \
  --base <released_model> --benign_data <downstream_task.jsonl> --safety_data <buffer.jsonl> \
  --benign_format qa_text --safety_format qa_text \
  --mode relu --lookahead_step sign --mu 100 \
  --steps 2000 --lr 5e-5 --batch 4 --accum 8 --scheduler linear --out <out_dir>
```

### Project Structure

```
LookAhead/
├── defense/
│   ├── lookahead_trainer.py   # LookAhead Defense
│   └── safety_buffer/         # Safety Buffer construction
├── attack/                    # Agentic FAB
├── common/                    # evaluation helpers
└── figs/
```

---

## 📝 Citation

```bibtex

```
