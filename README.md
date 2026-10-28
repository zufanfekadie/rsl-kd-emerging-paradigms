# Bounded Recursive Self-Learning & Knowledge Distillation

An applied research implementation combining **Bounded Recursive Self-Learning (RSL)** with holdout gating and **Knowledge Distillation (KD)** for efficient model compression.

---

## 1. System Architecture

```
   [15% Seed Data]
          │
          ▼
   Teacher v0 (Large MLP, ~60k params)
          │
          ├──► Recursive Self-Learning Loop
          │      1. Generate: sample pseudo-labels on 55% unlabeled pool
          │      2. Verify: check against independent external oracle
          │      3. Curate: keep high-confidence verified samples
          │      4. Train Copy: update candidate model clone
          │      5. Evaluate: test on protected 15% validation holdout
          │      6. Gate: accept if gain > ε (+0.5%), else rollback
          │
   Teacher Final (Improved, accurate, high-capacity)
          │
          ▼
   Knowledge Distillation Pipeline
          ├──► Hard-Label Student (2.4k params, trained only on hard labels)
          └──► Distilled Student (2.4k params, trained on Teacher Soft Targets, T=3.5, α=0.3)
          │
          ▼
   Untouched Benchmark (15% Final Test Set)
          - Accuracy
          - Parameter compression ratio
          - CPU inference latency (ms/100 samples)
```

---

## 2. Mathematical Formulations

### Combined Distillation Loss (Hinton et al., 2015)
$$\mathcal{L}_{\text{KD}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + (1 - \alpha) \cdot T^2 \cdot \mathcal{L}_{\text{KL}}(\sigma(z_t / T) \parallel \sigma(z_s / T))$$
- $\alpha = 0.3$: Weight between hard cross-entropy and soft teacher imitation.
- $T = 3.5$: Temperature to expose dark knowledge in logits.
- $T^2$: Rescales gradients to remain invariant as $T$ changes.

### Bounded Acceptance Gate
$$\text{Decision} = 
\begin{cases} 
\text{ACCEPT Checkpoint}, & \text{if } \text{Acc}_{\text{val}}(\mathcal{M}_{\text{proposal}}) \ge \text{Acc}_{\text{val}}(\mathcal{M}_{\text{current}}) + 0.005 \\
\text{ROLLBACK}, & \text{otherwise}
\end{cases}$$

---

## 3. How to Run

```bash
python main.py
```

### Run Benchmarks & Visualizations:
```bash
# Multi-seed evaluation suite
python experiments/multi_seed.py

# Temperature and alpha ablation study
python experiments/ablation.py

# Visualize dark knowledge in teacher logits
python experiments/dark_knowledge_viz.py

# Run fast interactive presentation demo
python demo.py
```

Results and publication-ready figures are saved in `artifacts/figures/`.
