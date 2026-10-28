# Bounded Recursive Self-Learning & Knowledge Distillation

Applied implementation for **iCog Labs AI Internship Training: Emerging AI Paradigms**.
Combines **Track A (Bounded Recursive Self-Learning Loop)** and **Track B (Knowledge Distillation)** into an integrated stretch-goal system.

---

## 1. System Architecture

```
   [15% Seed Data]
          │
          ▼
   Teacher v0 (Large MLP, ~100k params)
          │
          ├──► Recursive Self-Learning Loop (Track A)
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
   Knowledge Distillation Pipeline (Track B)
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
cd D:\projects\rsl_kd_project
python main.py
```

Results and charts will be generated inside `D:\projects\rsl_kd_project\artifacts\figures/`.
