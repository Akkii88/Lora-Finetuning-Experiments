# LLM-Parameter-Efficient-Fine-Tuning-with-LoRA

A comprehensive research project exploring Parameter-Efficient Fine-Tuning (PEFT) methods including LoRA, IA3, and Prompt Tuning for sentiment classification on DistilBERT.

---

## 📊 Project Overview

| Aspect | Details |
|--------|---------|
| **Base Model** | DistilBERT-base-uncased (66M parameters) |
| **Task** | Binary Sentiment Classification (IMDB Movie Reviews) |
| **Datasets** | IMDB (in-domain), Yelp Polarity (cross-domain) |
| **PEFT Methods** | LoRA, IA3, Prompt Tuning |

This research demonstrates how PEFT strategies enable efficient fine-tuning with minimal computational resources while achieving competitive accuracy.

---

## 🔬 Methods Comparison

### LoRA (Low-Rank Adaptation)
- **Trainable Parameters:** 739,586 (1.1% of total)
- **Accuracy:** 88.7%
- **Approach:** Adds low-rank matrices to attention weights

### IA3 (Infused Adapter by Inhibiting/Amplifying Activations)
- **Trainable Parameters:** 605,954 (0.9% of total)
- **Accuracy:** 86.2%
- **Approach:** Scales inner activations

### Prompt Tuning
- **Trainable Parameters:** 15,360 (0.02% of total)
- **Accuracy:** 82.5%
- **Approach:** Learns virtual tokens

---

## 📈 Experimental Results

### In-Domain Performance (IMDB)

| Method | Accuracy | Trainable Params | Training Memory |
|-------|----------|------------------|-----------------|
| Base (Untrained) | 50.0% | 0 | - |
| Prompt Tuning | 82.5% | 15,360 | 0.6 GB |
| IA3 | 86.2% | 605,954 | 0.7 GB |
| **LoRA (r=8)** | **88.7%** | **739,586** | **0.8 GB** |

### Hyperparameter Scaling (LoRA Rank)

| Rank | Accuracy | Parameters (K) |
|------|----------|---------------|
| r=4 | 85.3% | 370K |
| r=8 | 88.7% | 740K |
| r=16 | 88.9% | 1,480K |

### Data Efficiency (LoRA)

| Training Samples | Accuracy |
|------------------|----------|
| 100 | 70.0% |
| 500 | 87.6% |
| 1,000 | 88.7% |

### Cross-Domain Generalization (Yelp)

| Method | IMDB (In-Domain) | Yelp (Cross-Domain) | Δ |
|--------|------------------|---------------------|---|
| Base | 52.3% | 48.7% | -3.6% |
| Prompt Tuning | 87.2% | 85.3% | -1.9% |
| LoRA | 89.1% | 87.2% | -1.9% |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NLP Pipeline with PEFT                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Text → Tokenizer → Token IDs + Attention Mask                  │
│                           ↓                                          │
│                    DistilBERT Frozen                                 │
│                           ↓                                          │
│         ┌───────────────┴───────────────┐                           │
│         ↓                               ↓                            │
│   LoRA Adapter                   Prompt Tuning                       │
│   739,586 params                 15,360 params                     │
│         ↓                               ↓                            │
│         └───────────────┬───────────────┘                           │
│                         ↓                                            │
│               Linear Head + Loss                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Lora-Finetuning-Experiments/
├── finetune.py              # LoRA fine-tuning script
├── ia3_tuning.py           # IA3 fine-tuning script
├── prompt_tuning.py        # Prompt tuning script
├── compare_methods.py       # Compare all PEFT methods
├── cross_domain_eval.py    # Cross-domain evaluation (Yelp)
├── data_efficiency.py      # Data efficiency experiments
├── hyperparam_scaling.py   # LoRA rank scaling experiments
├── validate.py             # Model validation script
├── create_figures.py       # Generate academic figures
├── dashboard/              # Web dashboard
│   ├── backend/app.py      # Flask API
│   └── frontend/           # React frontend
├── figures/                # Generated figures
│   ├── fig1_memory_storage.png
│   ├── fig2_lora_architecture.png
│   ├── fig3_pipeline_flowchart.png
│   ├── fig4_rank_accuracy.png
│   ├── fig5_learning_curve.png
│   ├── fig6_domain_shift.png
│   └── fig7_decision_flowchart.png
└── requirements.txt         # Dependencies
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/Akkii88/Lora-Finetuning-Experiments.git
cd Lora-Finetuning-Experiments

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

```bash
# Fine-tune with LoRA
python finetune.py

# Fine-tune with IA3
python ia3_tuning.py

# Fine-tune with Prompt Tuning
python prompt_tuning.py

# Compare all methods
python compare_methods.py

# Cross-domain evaluation
python cross_domain_eval.py

# Data efficiency experiments
python data_efficiency.py

# Hyperparameter scaling
python hyperparam_scaling.py
```

### Running the Dashboard

```bash
# Start dashboard
bash run_dashboard.sh
```

Access the dashboard at: http://localhost:5002

---

## 🔑 Key Findings

1. **LoRA r=8 is optimal**: Provides best accuracy-to-parameters ratio
2. **Storage Efficiency**: 10x storage savings vs full fine-tuning
3. **Data Efficiency**: Good performance with as few as 500 samples
4. **Cross-Domain**: LoRA generalizes well to unseen domains

---

## 📝 LoRA Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LoRA Adaptation                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input (x) ──→ [W₀ frozen] ──→ Output (h)                      │
│      │             ↓                                             │
│      └────→ [A] → [B] → (α/r) ──→ ⊕                            │
│                                                                  │
│  Formula: h = W₀x + (α/r)BAx                                    │
│                                                                  │
│  Trainable: r(d+k) vs d×k (full)                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📄 License

MIT License

---

## 👤 Author

Ankit - [GitHub](https://github.com/Akkii88)

