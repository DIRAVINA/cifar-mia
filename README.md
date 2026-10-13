# Membership Inference Attacks on CIFAR-10

Implementation of **Shokri et al., "Membership Inference Attacks Against Machine Learning Models"** (IEEE S&P 2017, [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)) on CIFAR-10 with PyTorch, plus extra analysis, defenses (including differential privacy) and an interactive Streamlit dashboard.

The attack takes only black-box access to a trained model and decides whether a given image was in its training set. **Shadow models** copy the target's behavior on data whose membership is known, and **one attack model per class** learns to tell members from non-members from the output probabilities.

## Scope: what is and is not covered

This is a replication of the paper's **core attack and its CIFAR-10 experiments**, with extensions. It is not a full replication of every experiment in the paper.

| Paper component | Status |
|---|---|
| Shadow-model attack, per-class attack models, black-box setting | Implemented |
| CIFAR-10 CNN target, training-set-size sweep (Fig. 4) | Implemented |
| Train-test gap vs per-class precision (Fig. 11), member / non-member output distributions (Fig. 12 style) | Implemented |
| Top-k, rounding, temperature, L2 defenses (Table III) | Implemented (on CIFAR-10) |
| Extra: ROC / AUC with a no-shadow baseline, shadow-count sweep | Added |
| Extra: dropout, **DP-SGD** (Opacus, beyond the paper) | Added |
| Extra: interactive dashboard and live demo | Added |
| CIFAR-100, number-of-classes study, synthetic / noisy shadow data | Not implemented |
| Purchases, Locations, Texas hospital, MNIST, Adult datasets | Not implemented |
| Google Prediction API / Amazon ML targets | Not implemented (services unavailable) |

## Project structure

```
cifar-mia/
├── config.py                    # all settings (sizes, epochs, seeds, DP budgets)
├── run_experiments.py           # size sweep, output defenses, L2 -> results.json
├── run_analysis.py              # ROC/AUC, histograms, shadow-count sweep, gap vs precision
├── run_defense_experiments.py   # dropout and dropout + L2
├── run_dp_experiments.py        # DP-SGD at several privacy budgets
├── build_demo.py                # trains and saves models for the live demo
├── app.py                       # Streamlit dashboard
├── pages/1_Live_Demo.py         # live demo page
├── requirements.txt
├── src/
│   ├── data.py                  # load CIFAR, split target / shadow data
│   ├── model.py                 # CNN (2 conv + FC-128, Tanh, optional dropout)
│   ├── train.py                 # training and prediction functions
│   ├── shadow.py                # shadow models, attack training data
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

```bash
pip install -r requirements.txt
```

## How to run

Run from the project root with the environment active. Times are rough estimates on a CPU.

**To only view the dashboard**, the saved results are already in the repo:

```bash
streamlit run app.py
```

**To reproduce the experiments**, run in this order:

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python -m src.evaluate` | Full pipeline test: target, 10 shadow models, attack models. Caches shadow outputs (not tracked by git) | 15-25 min |
| 2 | `python run_experiments.py --exp all` | Size sweep, output defenses, L2 regularization | 1-2 h |
| 3 | `python run_analysis.py` | ROC / AUC, histograms, shadow-count sweep, gap vs precision | 15-30 min |
| 4 | `python run_defense_experiments.py` | Dropout, dropout + L2 | ~30 min |
| 5 | `python run_dp_experiments.py` | DP-SGD at epsilon 1, 5, 10, 50 | 1-1.5 h |
| 6 | `python build_demo.py` | Saves models for the live demo | 3-6 min |
| 7 | `streamlit run app.py` | Dashboard + Live Demo page | - |

Quick checks: `python -m src.data`, `python -m src.model`, `python -m src.train`, `python -m src.dp`.

## Dashboard and live demo

- **Overview**: key numbers and an attack pipeline diagram
- **Why the attack works**: member vs non-member confidence histograms, ROC curves against a no-shadow baseline, gap vs precision per class
- **Attack strength**: training-set size, number of shadow models
- **Defenses**: output defenses, L2, dropout, DP-SGD, and a privacy-vs-utility chart
- **Per-class results**: heatmap and drill-down
- **Live Demo** page: random images are picked automatically; the attacker gives a member / non-member verdict for each, the answers can be revealed afterwards, and a batch test shows the histogram of member vs non-member scores

## Results

Setup: CIFAR-10, target trained on 5,000 images; the evaluation uses 5,000 members and 5,000 non-members, so random guessing = 0.5. The baseline target reaches train accuracy 1.000 and test accuracy 0.560 (a gap of 0.44). All results are from a **single run (one seed)**.

### Attack quality (no defense)

| Metric | Value |
|---|---|
| Precision | 0.795 |
| Recall | 0.976 |
| AUC | 0.903 |
| AUC of a simple confidence-threshold baseline (no shadow models) | 0.865 |
| True-positive rate at 1% false-positive rate | 0.117 |
| True-positive rate at 0.1% false-positive rate | 0.007 |
| Correlation of per-class train-test gap and attack precision | 0.94 |

### Training-set size

| Train size | Target test acc | Attack precision | Attack recall |
|---|---|---|---|
| 2,500 | 0.482 | 0.816 | 0.974 |
| 5,000 | 0.560 | 0.793 | 0.976 |
| 10,000 | 0.592 | 0.770 | 0.982 |
| 15,000 | 0.620 | 0.752 | 0.978 |

### Number of shadow models

| Shadow models | Precision | Recall | AUC |
|---|---|---|---|
| 1 | 0.775 | 0.988 | 0.894 |
| 2 | 0.790 | 0.980 | 0.899 |
| 5 | 0.796 | 0.972 | 0.900 |
| 10 | 0.795 | 0.973 | 0.903 |

### Defenses

| Defense | Target test acc | Attack precision | Attack recall |
|---|---|---|---|
| None | 0.560 | 0.795 | 0.976 |
| Top k = 3 | 0.560 | 0.798 | 0.972 |
| Top k = 1 | 0.560 | 0.789 | 0.977 |
| Top k = 1, label only | 0.560 | 0.641 | 1.000 |
| Rounding d = 3 | 0.560 | 0.795 | 0.973 |
| Rounding d = 1 | 0.559 | 0.782 | 0.986 |
| Temperature t = 5 | 0.560 | 0.796 | 0.957 |
| Temperature t = 20 | 0.560 | 0.786 | 0.890 |
| L2, lambda = 1e-4 | 0.555 | 0.796 | 0.977 |
| L2, lambda = 1e-3 | 0.560 | 0.773 | 0.967 |
| L2, lambda = 3e-3 | 0.544 | 0.728 | 0.925 |
| L2, lambda = 1e-2 | 0.551 | 0.634 | 0.816 |
| Dropout 0.2 | 0.558 | 0.765 | 0.967 |
| Dropout 0.5 | 0.542 | 0.713 | 0.910 |
| Dropout 0.5 + L2 1e-3 | 0.536 | 0.706 | 0.903 |
| **DP-SGD, epsilon = 50** | 0.459 | 0.513 | 0.580 |
| **DP-SGD, epsilon = 10** | 0.418 | 0.512 | 0.527 |
| **DP-SGD, epsilon = 5** | 0.391 | 0.509 | 0.503 |
| **DP-SGD, epsilon = 1** | 0.318 | 0.510 | 0.493 |

In every defense experiment the attacker is assumed to know the defense: shadow models are trained or queried the same way as the target.

## Key findings

- **The attack works** (precision about 0.8, recall about 0.98) because the target overfits: members get near-certain predictions and non-members do not. Classes with a larger train-test gap are attacked more precisely (correlation 0.94).
- **Bigger training sets weaken the attack** because they overfit less. This matches the paper.
- **More shadow models help only slightly**: most of the benefit arrives by about 5 shadows.
- **The shadow attack beats a simple confidence threshold**, but the overall AUC gain is small (0.903 vs 0.865). The advantage is larger at low false-positive rates, but even there the attack finds few members (11.7% at a 1% false-positive rate).
- **Output-side defenses barely help.** Only releasing the label alone reduces precision noticeably (to 0.641), and the attack still beats guessing.
- **Regularization helps in proportion to its strength.** L2 at 1e-2 lowers precision to 0.634 at almost no accuracy cost in this setup; the attack is still above chance.
- **DP-SGD reduces the attack to chance** (precision about 0.51) at every tested epsilon, but costs 10 to 24 points of test accuracy.

## Differences from the paper

- Adam instead of SGD, 20 training epochs instead of up to 100, so models overfit quickly on a CPU.
- 10 shadow models for the main attack (the paper uses 100 on CIFAR), and 3 for the sweeps.
- Defenses were tested on CIFAR-10 (the paper reports Table III on the Purchase and Texas hospital datasets).
- DP-SGD, dropout, ROC analysis and the dashboard are extensions beyond the paper. The paper only discusses differential privacy in Section VIII without training such a model.

## Limitations

- Single seed per experiment: differences of about 0.02 or less are within noise.
- The attack's low true-positive rate at very low false-positive rates means that overall precision overstates the risk for any single record.
- DP-SGD was run with untuned settings (Adam, clipping norm 1.0, 20 epochs); a tuned run could keep more accuracy. The attack failing against DP models is an empirical result, not a proof of privacy.

## References

- R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P, 2017.
- M. Abadi et al. *Deep Learning with Differential Privacy.* ACM CCS, 2016.
- A. Yousefpour et al. *Opacus: User-Friendly Differential Privacy Library in PyTorch.* 2021.