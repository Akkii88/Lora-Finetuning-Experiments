# Lora-Finetuning-Experiments

A comprehensive research project exploring Parameter-Efficient Fine-Tuning (PEFT) methods including LoRA, IA3, and Prompt Tuning for sentiment classification on DistilBERT.

---

## 📊 Project Overview

| Aspect | Details |
|--------|---------|
| **Base Model** | DistilBERT-base-uncased (66M parameters) |
| **Task** | Binary Sentiment Classification (IMDB Movie Reviews) |
| **Datasets** | IMDB (in-domain), Yelp Polarity (cross-domain) |
| **PEFT Methods** | LoRA, IA3, Prompt Tuning |

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

| Method | Accuracy | Trainable Params |
|-------|----------|------------------|
| Base (Untrained) | 50.0% | 0 |
| Prompt Tuning | 82.5% | 15,360 |
| IA3 | 86.2% | 605,954 |
| **LoRA (r=8)** | **88.7%** | **739,586** |

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

| Method | IMDB | Yelp | Δ |
|--------|------|------|---|
| Base | 52.3% | 48.7% | -3.6% |
| Prompt Tuning | 87.2% | 85.3% | -1.9% |
| LoRA | 89.1% | 87.2% | -1.9% |

---

## 📁 Repository Structure

```
Lora-Finetuning-Experiments/
├── finetune.py              # LoRA fine-tuning script
├── ia3_tuning.py           # IA3 fine-tuning script
├── prompt_tuning.py        # Prompt tuning script
├── compare_methods.py       # Compare all PEFT methods
├── cross_domain_eval.py    # Cross-domain evaluation
├── data_efficiency.py      # Data efficiency experiments
├── hyperparam_scaling.py   # LoRA rank scaling
├── validate.py             # Model validation
├── create_figures.py       # Generate academic figures
├── dashboard/              # Web dashboard
├── figures/                # Generated figures
└── requirements.txt        # Dependencies
```

---

## 🚀 Getting Started

```bash
# Clone repository
git clone https://github.com/Akkii88/Lora-Finetuning-Experiments.git

# Install dependencies
pip install -r requirements.txt

# Run LoRA fine-tuning
python finetune.py

# Run IA3
python ia3_tuning.py

# Run Prompt Tuning
python prompt_tuning.py

# Start dashboard
bash run_dashboard.sh
```

---

## 🔑 Key Findings

1. **LoRA r=8 is optimal** - Best accuracy-to-parameters ratio
2. **Storage Efficiency** - 10x savings vs full fine-tuning
3. **Data Efficiency** - Good performance with 500+ samples
4. **Cross-Domain** - LoRA generalizes well to unseen domains

---

## 📄 License

MIT License

---

## 👤 Author

Ankit - [GitHub](https://github.com/Akkii88)
