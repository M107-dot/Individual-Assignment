# Neural Ordinary Differential Equations

**MLNN Tutorial — University of Hertfordshire 2025**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

---

## Overview

A residual network computes h_{t+1} = h_t + f(h_t; θ_t). As layers → ∞ and step size → 0, this becomes a differential equation: **dh/dt = f(h(t), t; θ)**. A Neural ODE parameterises this derivative with a neural network and uses a black-box ODE solver to integrate forward — giving infinite depth with shared parameters.

**Technique:** Neural ODEs — continuous depth, adjoint backprop, latent ODEs  
**Datasets:** Synthetic spiral + irregular time series  
**Difficulty:** Advanced / beyond-course — differential equations, Pontryagin adjoint

---

## What You Will Learn

- The **continuous limit** of residual networks — Neural ODEs as infinite-depth ResNets
- **ODE integration** as a forward pass — Euler, RK4, adaptive solvers
- **The adjoint method** — O(1) memory backpropagation via Pontryagin's principle
- **Continuous feature transformation** — how the ODE warps the data manifold
- **Latent ODEs** for irregularly-sampled time series (Rubanova et al. 2019)
- **FFJORD** — continuous normalising flows with Neural ODEs

---

## Repository Contents

```
neural-ode-tutorial/
├── node_tutorial.ipynb          ← Full Jupyter notebook
├── NeuralODE_Tutorial.docx      ← Tutorial document (Word)
├── figures_node/                ← All generated figures (PNG, 180 dpi)
│   ├── fig1_resnet_vs_node.png
│   ├── fig2_trajectories.png
│   ├── fig3_adjoint_method.png
│   ├── fig4_irregular_timeseries.png
│   └── fig5_comparison_applications.png
├── README.md
└── LICENSE
```

---

## Quick Start

```bash
git clone https://github.com/yourusername/neural-ode-tutorial.git
cd neural-ode-tutorial
pip install torch numpy matplotlib scikit-learn scipy notebook
jupyter notebook node_tutorial.ipynb
```

Expected runtime: ~3 minutes.

---

## Key Equations

| Component | Equation | Meaning |
|-----------|----------|---------|
| Forward ODE | dh/dt = f_θ(h(t), t) | Dynamics defined by NN |
| Solution | h(T) = h(0) + ∫₀ᵀ f_θ dt | Integrate forward |
| Adjoint | da/dt = −aᵀ ∂f/∂h | Reverse-time gradient |
| Param grad | dL/dθ = −∫ aᵀ ∂f/∂θ dt | Gradient via adjoint |

---

## Comparison: ResNet vs Neural ODE

| Property | ResNet | Neural ODE |
|----------|--------|------------|
| Parameters | O(depth × size) | O(size) shared |
| Memory | O(T) | O(1) adjoint |
| Time series | Fixed steps only | Irregular natural |
| Depth | Fixed | Adaptive |

---

## Accessibility

- Okabe-Ito colourblind-safe palette; line-style dual encoding
- Alt-text on all figures; H1→H2 screen-reader hierarchy
- All maths spelled out in plain English alongside symbolic form

---

## References

1. Chen et al. (2018) — Neural ordinary differential equations. https://arxiv.org/abs/1806.07366
2. Rubanova et al. (2019) — Latent ODEs for irregularly-sampled time series. https://arxiv.org/abs/1907.03907
3. Grathwohl et al. (2019) — FFJORD. https://arxiv.org/abs/1810.01367
4. Dupont et al. (2019) — Augmented Neural ODEs. https://arxiv.org/abs/1904.01681
5. Pontryagin et al. (1962) — The Mathematical Theory of Optimal Processes. Wiley.

---

## Licence

MIT — see [LICENSE](LICENSE).

*University of Hertfordshire · MLNN Assignment 2025*
