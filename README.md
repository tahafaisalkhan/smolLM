# 🚀 SmolLM: From Scratch to DPO-Tuned Language Model

This repository contains a 3-phase implementation and enhancement pipeline for the **SmolLM** architecture, a compact transformer model inspired by LLaMA3. The project consists of:

1. **Implementing the Base Language Model (SmolLM)**
2. **Applying Parameter-Efficient Fine-Tuning via LoRA**
3. **Enhancing with Direct Preference Optimization (DPO)**

---

## 🧱 1. SmolLM - A Tiny Transformer

SmolLM is a transformer-based language model inspired by LLaMA3. This implementation builds the model from the ground up using:

- 🧮 **RMS Norm** ([Zhang & Sennrich, 2019])
- 🌀 **Rotary Positional Embeddings (RoPE)** ([Su et al., 2023])
- 🎯 **Grouped Query Attention (GQA)** ([Ainslie et al., 2023])
- ⚙️ **SwiGLU Feedforward Networks** ([Shazeer, 2020])
- 🧠 **Weight Tied Embedding & Projection Head**

> 📚 See the `CS5302_PA2_manual.pdf` for architectural details.

### 🏋️ Training Stats

| Epoch | Train Loss | Train Perplexity | Val Loss | Val Perplexity |
|-------|------------|------------------|----------|----------------|
| 1     | 3.4935     | 32.90            | 3.2364   | 25.44          |
| 2     | 3.2192     | 25.01            | 3.1136   | 22.50          |
| 3     | 3.0998     | 22.19            | 3.0835   | 21.83          |
| 4     | 3.0436     | 20.98            | 3.0657   | 21.45          |
| 5     | 2.9960     | 20.00            | 3.0609   | 21.35          |

---

## 🔧 2. LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) was applied for efficient fine-tuning using just **0.90%** of the original model's parameters (1.2M vs 135M).

### 🧬 LoRA Configurations

- Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`
- Trainable Parameters:
  - Total: **1,221,120**
  - Full Model: **135,736,128**

### 📉 Perplexity Results

| Prompt                                        | Fully Finetuned | LoRA Tuned |
|----------------------------------------------|------------------|------------|
| Once upon a time, in a distant galaxy,        | 5.9674           | 5.9229     |
| The future of artificial intelligence is      | 22.8368          | 17.3586    |
| A wise old wizard once said,                 | 53.2328          | 20.4430    |

LoRA achieves significant gains with minimal compute cost!

---

## 🎯 3. Direct Preference Optimization (DPO)

DPO was used to further align the model to human preferences using pairwise ranking signals.

### 📈 DPO Training Overview

- Initial Loss: `0.6931`
- Final Running Loss: **`0.4578`**
- Accuracy:
  - Reference Model: **72.0%**
  - DPO-Tuned Policy Model: **93.0%**

> The logits difference curve shows consistent alignment improvements throughout training.
![asd](https://github.com/user-attachments/assets/3a03fbfd-1e96-4bbd-bbe2-e3067aae9647)
![output](https://github.com/user-attachments/assets/9f4831f6-46ba-455f-af9b-67d207aac5a8)


---

## 📂 Project Structure
📁 smollm/
├── layers.py # RMSNorm, SwiGLU, decoder block
├── attention.py # RoPE and GQA
├── model.py # SmolModel and SmolLM
├── train.py # Training loop
├── lora_finetune.py # LoRA integration
├── dpo_train.py # Direct Preference Optimization


---

## 📚 References

- [SmolLM Paper (Allal et al., 2024)](https://arxiv.org/abs/2404.xxxxx)
- [RMS Norm (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)
- [RoPE (Su et al., 2023)](https://arxiv.org/abs/2104.09864)
- [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)
- [SwiGLU (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)

---

## 🧪 Results Summary

| Phase     | Loss ↓  | Perplexity ↓ | Accuracy ↑ | Params Trained ↓ |
|-----------|---------|--------------|------------|------------------|
| Base LM   | 2.9960  | 20.00        | -          | 100%             |
| LoRA      | -       | 17.36 (↓20%) | -          | 0.90%            |
| DPO       | 0.4578  | -            | 93.0%      | N/A              |

---
