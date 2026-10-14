# Membership Inference Attacks on CIFAR-10

Implementation of **Shokri et al., "Membership Inference Attacks Against Machine Learning Models"** (IEEE S&P 2017, [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)) on CIFAR-10 with PyTorch, plus extra analysis, defenses (including differential privacy) and an interactive Streamlit dashboard.

The attack takes only black-box access to a trained model and decides whether a given image was in its training set. **Shadow models** copy the target's behavior on data whose membership is known, and **one attack model per class** learns to tell members from non-members from the output probabilities.

**Full end-to-end report** (problem statement, paper summary, design, all results, verification and audit, reproduction): [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)

## Scope: what is and is not covered

This is a replication of the paper's **core attack and its CIFAR-10 experiments**, with extensions. It is not a full replication of every experiment in the paper.

| Paper component | Status |
|---|---|
| Shadow-model attack, per-class attack models, black-box setting | Implemented |
| CIFAR-10 CNN target, training-set-size sweep (Fig. 4) | Implemented, compared number by number |
| Train-test gap vs per-class precision (Fig. 11), member / non-member output distributions (Fig. 12 style) | Implemented |
| Top-k, rounding, temperature, L2 defenses (Table III) | Implemented (on CIFAR-10) |
| Extra: ROC / AUC with a no-shadow baseline, TPR at low FPR, shadow-count sweep | Added |
| Extra: repeated runs over 5 seeds (mean ± std) | Added |
| Extra: dropout, **DP-SGD** (Opacus, beyond the paper) | Added |
| Extra: interactive dashboard and live demo | Added |
| CIFAR-100, number-of-classes study, synthetic / noisy shadow data | Not implemented |
| Purchases, Locations, Texas hospital, MNIST, Adult datasets | Not implemented |
| Google Prediction API / Amazon ML targets | Not implemented (services unavailable) |

## Project structure

```
cifar-mia/
├── config.py                    # all settings (sizes, epochs, seeds, DP budgets)
├── run_all.py                   # runs the whole pipeline in order
├── run_experiments.py           # size sweep, output defenses, L2 -> results.json
├── run_analysis.py              # ROC/AUC, histograms, shadow-count sweep, gap vs precision
├── run_defense_experiments.py   # dropout and dropout + L2
├── run_dp_experiments.py        # DP-SGD at several privacy budgets
├── run_seeds.py                 # baseline + output defenses over 5 seeds
├── build_demo.py                # trains and saves models for the live demo
├── app.py                       # Streamlit dashboard
├── pages/1_Live_Demo.py         # live demo page
├── requirements.txt
├── docs/PROJECT_REPORT.md       # full end-to-end project report
├── src/
│   ├── data.py                  # load CIFAR, seeding, split target / shadow data
│   ├── model.py                 # CNN (2 conv + FC-128, Tanh, optional dropout)
│   ├── train.py                 # training (paper's L2 definition), prediction
│   ├── shadow.py                # shadow models, attack training data, shadow cache
│   ├── attack.py                # per-class attack models
│   ├── evaluate.py              # precision / recall, full pipeline test
│   ├── mitigations.py           # top-k, rounding, temperature
│   └── dp.py                    # DP-SGD training with Opacus
└── results/
    ├── results.json             # all saved experiment results (used by the app)
    └── demo/                    # small saved models for the live demo
```

## Setup

Requires Python 3.9+. A GPU is optional (everything runs on CPU).

```bash
git clone https://github.com/DIRAVINA/cifar-mia.git
cd cifar-mia
python -m venv venv
```

Activate the environment:

- Windows PowerShell: `venv\Scripts\activate`
- Mac / Linux: `source venv/bin/activate`

For an NVIDIA GPU, first install the CUDA build of PyTorch (see [pytorch.org](https://pytorch.org)), e.g.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then:

```bash
pip install -r requirements.txt
```

## How to run

Run from the project root with the environment active.

**To only view the dashboard**, the saved results are already in the repo:

```bash
streamlit run app.py
```

**To reproduce everything** in one go:

```bash
python run_all.py
```

or step by step. Times measured on a laptop with an RTX 3050 (4 GB); a CPU-only machine takes about 3 to 5 times longer.

| Step | Command | What it does | Time (GPU) |
|---|---|---|---|
| 1 | `python -m src.evaluate` | Full pipeline test: target, 10 shadow models, attack models. Caches shadow outputs (not tracked by git) | ~3 min |
| 2 | `python run_experiments.py --exp all` | Size sweep, output defenses, L2 regularization | ~40 min |
| 3 | `python run_analysis.py` | ROC / AUC, histograms, shadow-count sweep, gap vs precision | ~10 min |
| 4 | `python run_defense_experiments.py` | Dropout, dropout + L2 | ~10 min |
| 5 | `python run_dp_experiments.py` | DP-SGD at epsilon 1, 5, 10, 50 | ~20 min |
| 6 | `python run_seeds.py` | Baseline and output defenses over 5 seeds | ~50 min |
| 7 | `python build_demo.py --force` | Target and attack models for each demo defense (none, label-only, L2, dropout, DP-SGD) | ~15 min |
| 8 | `streamlit run app.py` | Dashboard + Live Demo page | - |

Steps 2 to 7 only need step 1 and may run in parallel terminals (results are saved under a lock). Each needs about 2 GB of RAM: on an 8 GB laptop run at most 3 at once. `python run_seeds.py --seeds 3 4` runs only some seeds.

Quick checks: `python -m src.data`, `python -m src.model`, `python -m src.train`, `python -m src.dp`.

## Dashboard and live demo

- **Overview**: key numbers, run-to-run variation over 5 seeds, attack pipeline diagram
- **Why the attack works**: member vs non-member confidence histograms, ROC curves against a no-shadow baseline, gap vs precision per class
- **Attack strength**: training-set size (with the paper's Fig. 4 values), number of shadow models
- **Defenses**: output defenses, L2, dropout, DP-SGD, and a privacy-vs-utility chart
- **Per-class results**: heatmap and drill-down
- **Live Demo** page: random images are picked automatically; the attacker gives a member / non-member verdict for each. A **Defense** dropdown (none, label-only, L2, dropout, DP-SGD) re-scores the same images, so you can watch the attack drop toward a coin flip under DP-SGD. Answers can be revealed, and a batch test shows the member vs non-member score histogram

## Results

Setup: CIFAR-10, target trained on 5,000 images; the evaluation uses 5,000 members and 5,000 non-members, so random guessing = 0.5. The baseline target reaches train accuracy 1.000 and test accuracy 0.565 (a gap of 0.435). Main numbers are from one run (seed 42); run-to-run variation is measured over 5 seeds below.

### Attack quality (no defense)

| Metric | Value |
|---|---|
| Precision | 0.790 |
| Recall | 0.984 |
| Attack accuracy | 0.861 |
| Median per-class precision (paper Fig. 4: 0.74) | 0.762 |
| AUC | 0.901 |
| AUC of a simple confidence-threshold baseline (no shadow models) | 0.856 |
| True-positive rate at 1% false-positive rate | 0.114 (baseline 0.015) |
| True-positive rate at 0.1% false-positive rate | 0.009 (baseline 0.001) |
| Correlation of per-class train-test gap and attack precision | 0.96 |

### Run-to-run variation (5 seeds)

Each seed draws a new data split and trains a new target, 10 new shadow models and new attack models.

| Defense | Precision | Recall | Attack accuracy |
|---|---|---|---|
| None | 0.792 ± 0.007 | 0.976 ± 0.006 | 0.860 ± 0.005 |
| Top k = 3 | 0.793 ± 0.007 | 0.976 ± 0.006 | 0.861 ± 0.005 |
| Top k = 1 | 0.785 ± 0.008 | 0.979 ± 0.005 | 0.855 ± 0.007 |
| Top k = 1, label only | 0.641 ± 0.003 | 1.000 ± 0.000 | 0.721 ± 0.004 |
| Rounding d = 3 | 0.792 ± 0.007 | 0.976 ± 0.005 | 0.860 ± 0.005 |
| Rounding d = 1 | 0.779 ± 0.009 | 0.986 ± 0.003 | 0.853 ± 0.007 |
| Temperature t = 5 | 0.785 ± 0.007 | 0.963 ± 0.006 | 0.850 ± 0.005 |
| Temperature t = 20 | 0.774 ± 0.006 | 0.912 ± 0.018 | 0.823 ± 0.007 |

No-defense AUC 0.900 ± 0.005 (confidence baseline 0.855 ± 0.008), TPR at 1% FPR 0.117 ± 0.014, target test accuracy 0.559 ± 0.007. Differences in precision below about 0.015 are within noise.

### Training-set size (paper, Fig. 4)

| Train size | Target test acc | Attack precision | Attack recall | Median class precision | Paper (median) |
|---|---|---|---|---|---|
| 2,500 | 0.502 | 0.817 | 0.962 | 0.814 | 0.78 |
| 5,000 | 0.565 | 0.786 | 0.985 | 0.758 | 0.74 |
| 10,000 | 0.595 | 0.765 | 0.984 | 0.730 | 0.72 |
| 15,000 | 0.624 | 0.747 | 0.983 | 0.718 | 0.71 |

### Number of shadow models

| Shadow models | Precision | Recall | AUC |
|---|---|---|---|
| 1 | 0.766 | 0.992 | 0.893 |
| 2 | 0.780 | 0.988 | 0.896 |
| 5 | 0.791 | 0.981 | 0.897 |
| 10 | 0.790 | 0.984 | 0.901 |

### Defenses

L2 lambda is the paper's definition: `loss + lambda * sum(theta^2)`.

| Defense | Target test acc | Attack accuracy | Attack precision | Attack recall |
|---|---|---|---|---|
| None | 0.565 | 0.861 | 0.790 | 0.984 |
| Top k = 3 | 0.565 | 0.862 | 0.791 | 0.983 |
| Top k = 1 | 0.565 | 0.854 | 0.781 | 0.984 |
| Top k = 1, label only | 0.565 | 0.717 | 0.639 | 1.000 |
| Rounding d = 3 | 0.565 | 0.861 | 0.790 | 0.983 |
| Rounding d = 1 | 0.565 | 0.849 | 0.773 | 0.988 |
| Temperature t = 5 | 0.565 | 0.850 | 0.781 | 0.973 |
| Temperature t = 20 | 0.565 | 0.821 | 0.771 | 0.915 |
| L2, lambda = 1e-4 | 0.565 | 0.856 | 0.783 | 0.984 |
| L2, lambda = 1e-3 | 0.542 | 0.793 | 0.738 | 0.909 |
| L2, lambda = 3e-3 | 0.525 | 0.738 | 0.691 | 0.860 |
| L2, lambda = 5e-3 | 0.538 | 0.674 | 0.637 | 0.812 |
| L2, lambda = 1e-2 | 0.547 | 0.565 | 0.552 | 0.687 |
| Dropout 0.2 | 0.557 | 0.826 | 0.762 | 0.947 |
| Dropout 0.5 | 0.550 | 0.770 | 0.708 | 0.920 |
| Dropout 0.5 + L2 1e-3 | 0.539 | 0.732 | 0.683 | 0.866 |
| **DP-SGD, epsilon = 50** | 0.458 | 0.519 | 0.515 | 0.638 |
| **DP-SGD, epsilon = 10** | 0.415 | 0.513 | 0.510 | 0.624 |
| **DP-SGD, epsilon = 5** | 0.388 | 0.514 | 0.512 | 0.590 |
| **DP-SGD, epsilon = 1** | 0.321 | 0.511 | 0.510 | 0.587 |

In every defense experiment the attacker is assumed to know the defense: shadow models are trained or queried the same way as the target. L2, dropout and DP-SGD rows use 3 shadow models.

## Key findings

- **The attack works** (precision 0.79, recall 0.98) because the target overfits: 98% of members get at least 0.95 confidence in their true class, against 27% of non-members. Classes with a larger train-test gap are attacked more precisely (correlation 0.96).
- **The paper's CIFAR-10 numbers are reproduced**: median per-class precision 0.81 / 0.76 / 0.73 / 0.72 vs the paper's 0.78 / 0.74 / 0.72 / 0.71, and bigger training sets weaken the attack.
- **More shadow models help only slightly**: one shadow already gives most of the attack's power.
- **The shadow attack beats a simple confidence threshold** mostly at low false-positive rates (TPR at 1% FPR 0.114 vs 0.015), but even there it exposes few individual members.
- **Output-side defenses barely help.** Only releasing the label alone reduces precision noticeably (to 0.639), and the attack still beats guessing: it calls every correctly classified image a member, so precision = 1 / (1 + test accuracy).
- **Regularization helps in proportion to its strength.** L2 at 1e-2 lowers attack accuracy to 0.565 for 1.8 points of test accuracy.
- **DP-SGD reduces the attack to chance** (precision about 0.51) at every tested epsilon, but costs 11 to 24 points of test accuracy.

## Differences from the paper

- Adam instead of SGD, 20 training epochs instead of up to 100 (the models already reach 100% training accuracy).
- 10 shadow models for the main attack (the paper uses 100 on CIFAR), and 3 for the sweeps.
- Defenses were tested on CIFAR-10 (the paper reports Table III on the Purchase and Texas hospital datasets).
- DP-SGD, dropout, ROC analysis, seed repeats and the dashboard are extensions beyond the paper. The paper only discusses differential privacy in Section VIII without training such a model.

## Verification and fixes

The project was audited end to end; see [Section 9 of the report](docs/PROJECT_REPORT.md#9-verification-and-audit). The originally committed results were confirmed genuine (all saved models reproduce their numbers exactly). Fixes in the final version:

- **L2 now matches the paper's lambda.** Adam's `weight_decay=lambda` equals a penalty of only `lambda/2 * sum(theta^2)`, so earlier L2 results were at half the stated strength (old "1e-2" = new 5e-3, both precision about 0.635).
- TPR at low FPR uses the standard definition (highest TPR with FPR at or below the limit) instead of interpolation.
- Attack models are seeded per class; seeding also covers Python, the GPU and cuDNN.
- The shadow cache records its settings and is retrained when `config.py` changes.
- Run-to-run variation is measured over 5 seeds.
- Median per-class precision is reported for a like-for-like comparison with the paper's Fig. 4.
- Faster training loops, attack models on CPU, lower memory use, and results saved per key so scripts can run in parallel.

## Limitations

- Training-defense sweeps (L2, dropout, DP-SGD) are single runs; use the seed standard deviations above as a guide to their noise.
- The attack's low true-positive rate at very low false-positive rates means that overall precision overstates the risk for any single record. Precision also assumes a 50/50 member ratio.
- DP-SGD was run with untuned settings (Adam, clipping norm 1.0, 20 epochs); a tuned run could keep more accuracy. The attack failing against DP models is an empirical result, not a proof of privacy.
- Results on other hardware differ by a few thousandths (floating-point order), well within the seed variation.

## References

- R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P, 2017.
- M. Abadi et al. *Deep Learning with Differential Privacy.* ACM CCS, 2016.
- A. Yousefpour et al. *Opacus: User-Friendly Differential Privacy Library in PyTorch.* 2021.
- S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF, 2018.
- N. Carlini et al. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022.
