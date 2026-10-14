# Membership Inference Attacks on CIFAR-10: End-to-End Project Report

Course project, Data Privacy and Security.
Replication and extension of R. Shokri, M. Stronati, C. Song, V. Shmatikov,
*Membership Inference Attacks Against Machine Learning Models*, IEEE Symposium on
Security and Privacy 2017 ([arXiv:1610.05820](https://arxiv.org/abs/1610.05820)).

Repository: <https://github.com/DIRAVINA/cifar-mia>

---

## Contents

1. [Summary](#1-summary)
2. [Problem statement](#2-problem-statement)
3. [The research paper](#3-the-research-paper)
4. [Scope of this project](#4-scope-of-this-project)
5. [System design](#5-system-design)
6. [Implementation](#6-implementation)
7. [Experimental setup](#7-experimental-setup)
8. [Results](#8-results)
9. [Verification and audit](#9-verification-and-audit)
10. [Discussion](#10-discussion)
11. [Limitations](#11-limitations)
12. [How to reproduce](#12-how-to-reproduce)
13. [Code reference](#13-code-reference)
14. [References](#14-references)
15. [Appendix: glossary](#15-appendix-glossary)

---

## 1. Summary

A machine learning model is trained on private data. Can someone who can only *query* the
model tell whether a particular record was part of that private training set? If yes, the
model leaks information about individuals: for a model trained on hospital records, knowing
that a person's record was in the training set reveals that the person was a patient.

This project implements the **shadow-model membership inference attack** of Shokri et al.
(2017) against a convolutional neural network trained on CIFAR-10, reproduces the paper's
CIFAR-10 findings, and extends the study with ROC/AUC analysis, a no-shadow baseline,
repeated runs over several seeds, two extra defenses (dropout and differentially private
training with DP-SGD), an interactive dashboard and a live demo.

**Main results** (CIFAR-10, target trained on 5,000 images, 5,000 members vs 5,000
non-members, so random guessing = 0.5):

| Question | Answer |
|---|---|
| Does the attack work? | Yes. Precision **0.792 ± 0.007**, recall **0.976 ± 0.006**, AUC **0.900 ± 0.005** (mean ± std over 5 seeds). |
| Do we reproduce the paper? | Yes. Median per-class precision 0.81 / 0.76 / 0.73 / 0.72 for 2,500 / 5,000 / 10,000 / 15,000 training images, vs the paper's 0.78 / 0.74 / 0.72 / 0.71 (Fig. 4). |
| Why does it work? | Overfitting: 100% train vs 56.5% test accuracy. Per-class train-test gap and attack precision correlate at 0.96. |
| Are shadow models needed? | They help mostly at low false-positive rates: TPR at 1% FPR 0.114 vs 0.015 for a plain confidence threshold (AUC 0.901 vs 0.856). |
| Do output defenses help? | Barely. Top-k, rounding, temperature: precision 0.77 to 0.79. Label-only: 0.639, still well above chance. |
| Does regularization help? | Yes, in proportion to strength. L2 lambda = 1e-2: attack accuracy 0.565, test accuracy 0.547 (vs 0.565). |
| Does differential privacy help? | DP-SGD (epsilon 1 to 50) brings the attack to chance (precision 0.510 to 0.515), at a cost of 11 to 24 points of test accuracy. |

The project was also audited end to end (Section 9): the originally committed results were
confirmed genuine, and 13 issues were fixed, most importantly an L2 regularization strength
that was half the paper's definition.

---

## 2. Problem statement

### 2.1 Setting

Following Section IV of the paper:

* A **target model** `f_target` is a classifier trained on a private dataset
  `D_train_target`. For any input `x` it returns a **prediction vector**: one probability
  (confidence) per class, summing to 1.
* The **attacker** has **black-box query access**: they can submit a record and read the
  prediction vector, nothing else. They do not see the model's weights or training data.
* The attacker knows the input/output format and either the model type and training
  algorithm, or has access to the same training service (e.g. an ML-as-a-service API).
* The attacker has some **background data** from the same population, **disjoint** from the
  target's training data.

### 2.2 The attack problem

> Given a labeled record `(x, y)` and black-box access to `f_target`, decide whether
> `(x, y)` was in `D_train_target` ("member", **in**) or not ("non-member", **out**).

This is a binary decision. The attacker succeeds if the decision is correct.

### 2.3 How success is measured

The evaluation set contains an equal number of members and non-members, so random guessing
has accuracy 0.5.

| Metric | Meaning |
|---|---|
| **Precision** | Of the records the attacker called members, the fraction that really are members. The paper's main metric. |
| **Recall** | Of the real members, the fraction the attacker found. |
| **Attack accuracy** | Fraction of all member/non-member decisions that are correct. |
| **AUC** (extension) | Area under the ROC curve of the attack's membership score; 0.5 = random, 1.0 = perfect. Threshold-free. |
| **TPR at low FPR** (extension) | Members found while wrongly accusing at most 1% (or 0.1%) of non-members. Measures how reliably *individual* records can be exposed. |

### 2.4 Project goals

1. Implement the shadow-model attack exactly as described in the paper and reproduce its
   CIFAR-10 results (Fig. 4: effect of training-set size).
2. Explain *why* the attack works (overfitting, Section VII and Fig. 11 of the paper).
3. Evaluate the paper's defenses (Table III: top-k, rounding, temperature, L2).
4. Extend the study: stronger analysis (ROC/AUC, a no-shadow baseline, seed variance) and
   stronger defenses (dropout, DP-SGD).
5. Make the results explorable: dashboard and live demo.

---

## 3. The research paper

### 3.1 Key idea

Models behave differently on data they were trained on than on data they have never seen:
typically they are more confident on training records. The attack *learns* this difference
with a supervised classifier, the **attack model**. The difficulty is that supervised
training needs labeled examples of "member" and "non-member" outputs, and the attacker does
not know the target's training set.

### 3.2 Shadow training

The paper's answer is **shadow models**:

1. The attacker trains `k` shadow models that imitate the target: same architecture and
   training procedure, trained on the attacker's own data.
2. For each shadow model the attacker *knows* the training set. They query the shadow model
   on its training records (label "in") and on other records (label "out").
3. These labeled prediction vectors form the training set of the attack model.
4. The attack model is then applied to the target's prediction vectors.

The attack is split into **one attack model per class**: the shape of a prediction vector
depends strongly on the true class, so a separate member/non-member classifier per class is
more accurate. The attack model's input is the prediction vector; the record's true label
selects which per-class model is used.

### 3.3 What the paper found (relevant to this project)

* CIFAR-10 CNN targets are highly vulnerable: median per-class attack precision 0.78, 0.74,
  0.72, 0.71 for training sets of 2,500, 5,000, 10,000, 15,000 images, recall almost 1
  (Fig. 4).
* Larger training sets reduce leakage (less overfitting).
* The per-class train-test accuracy gap is strongly related to attack precision (Fig. 11).
* Output defenses (top-k, rounding, temperature) barely help; even releasing only the
  predicted label leaks membership. L2 regularization helps but can cost accuracy
  (Table III, on the Purchase and Texas hospital datasets).

### 3.4 Paper settings used here

| Setting | Paper (CIFAR) | This project |
|---|---|---|
| Target architecture | 2 x (conv + max-pool), FC 128, softmax, Tanh | Same |
| Learning rate | 0.001 (decay 1e-7) | 0.001 (Adam, no decay) |
| Max epochs | 100 | 20 (already 100% train accuracy) |
| Target train sizes | 2,500 / 5,000 / 10,000 / 15,000 | Same |
| Target test set | Same size as train, disjoint | Same |
| Shadow models | 100 | 10 (main), 3 (sweeps) |
| Shadow data | Disjoint from target data, may overlap between shadows | Same |
| Attack model | Per class, FC 64 + ReLU + softmax | Same |
| Attacker knows defense | Shadows trained / queried like the target | Same |

---

## 4. Scope of this project

| Paper component | Status |
|---|---|
| Shadow-model attack, per-class attack models, black-box setting | Implemented |
| CIFAR-10 CNN target, training-set-size sweep (Fig. 4) | Implemented, compared number by number |
| Train-test gap vs per-class precision (Fig. 11), member / non-member output distributions (Fig. 12 style) | Implemented |
| Top-k, rounding, temperature, L2 defenses (Table III) | Implemented on CIFAR-10 |
| ROC / AUC, TPR at low FPR, no-shadow confidence baseline | Added |
| Number-of-shadow-models sweep | Added |
| Repeated runs over 5 seeds (mean ± std) | Added |
| Dropout and DP-SGD (Opacus) defenses | Added |
| Streamlit dashboard and live demo | Added |
| CIFAR-100, number-of-classes study, synthetic / noisy shadow data (Sections V-C, VI) | Not implemented |
| Purchases, Locations, Texas hospital, MNIST, Adult datasets | Not implemented |
| Google Prediction API / Amazon ML targets | Not possible (services discontinued) |

---

## 5. System design

### 5.1 Data split

CIFAR-10's 50,000 training and 10,000 test images are merged into one pool of 60,000 and
split with a fixed seed:

```
60,000 CIFAR-10 images (shuffled with seed 42)
├── target train   5,000   -> the "members" the attacker tries to detect
├── target test    5,000   -> the "non-members" (same size, so guessing = 0.5)
└── shadow pool   50,000   -> attacker's own data, disjoint from all target data
        each shadow model i: random 5,000 for training ("in"), another 5,000 for "out"
        (different shadows may reuse images, as in the paper)
```

For the training-set-size sweep, the target train/test sizes change and the shadow pool is
whatever remains (for 15,000 the pool is exactly 30,000 images).

### 5.2 Attack pipeline

```
 Attacker side                                           Victim side
 ─────────────                                           ───────────
 shadow pool ──► train k shadow models (same recipe as target)
                     │
                     │ query each shadow on its "in" and "out" images
                     ▼
         (prediction vector, true label, in/out) rows
                     │
                     │ split rows by true label
                     ▼
         10 attack models (one per class): vector ──► P(member)
                     ▲
                     │ prediction vector + true label
 record (x, y) ──────┴──────── query ───────────────► target model (black box)
                     │
                     ▼
         verdict: member if P(member) >= 0.5
```

### 5.3 Evaluation

The trained attack is applied to all 10,000 target records (5,000 members + 5,000
non-members). Precision, recall and accuracy are computed overall and per class; the
attack's membership score also gives the ROC curve and AUC.

### 5.4 Defenses

* **Output defenses** change what the API returns, not the model: top-k, label only,
  rounding, temperature. Applied to target *and* shadow outputs (the attacker knows the
  defense).
* **Training defenses** change how the model is trained: L2, dropout, DP-SGD. Target *and*
  shadow models are trained with the same defense.

---

## 6. Implementation

Python 3 and PyTorch. All settings live in `config.py`.

### 6.1 Data (`src/data.py`)
Downloads CIFAR-10 with torchvision, scales pixels to [0, 1], standardizes each channel
(mean 0, std 1) and returns `(60000, 3, 32, 32)` float arrays. `split_target_and_shadow_pool`
makes the disjoint split of Section 5.1 with a seeded permutation. `set_seed` seeds Python,
NumPy and PyTorch (CPU and GPU) and makes cuDNN deterministic.

### 6.2 Models (`src/model.py`, `src/attack.py`)
* `CifarCNN`: Conv(3→32, 5x5) → Tanh → MaxPool(2) → Conv(32→64, 5x5) → Tanh → MaxPool(2)
  → FC(1600→128) → Tanh → [optional Dropout] → FC(128→10). Outputs logits; softmax gives the
  prediction vector. 259,914 parameters.
* `AttackMLP`: FC(10→64) → ReLU → FC(64→2), one per class.

### 6.3 Training (`src/train.py`)
Adam, learning rate 0.001, batch size 64, 20 epochs, cross-entropy loss. Optional L2 with
the paper's definition: `loss = cross-entropy + lambda * sum(theta^2)` over all parameters.
Mini-batches are sliced directly from tensors on the device (`minibatches`).

### 6.4 Shadow models (`src/shadow.py`)
`train_shadow_models` trains `k` shadows, each on a random 5,000-image subset of the shadow
pool, and queries it on those 5,000 ("in") and on another disjoint 5,000 ("out"). Shadow
`i` uses seed `SEED + i`. The 10 main shadows are cached in
`results/shadow_attack_data.npz` together with the settings they were made with; the cache
is retrained automatically if `config.py` changes.

### 6.5 Attack (`src/attack.py`)
`train_attack_models` trains one `AttackMLP` per class (50 epochs, Adam, lr 0.001, batch 64)
on the shadow rows with that true label, with seed `SEED + class`. Training runs on the CPU
with one thread: for such a tiny network this is about 3x faster than a GPU.
`predict_membership` gives 0/1 verdicts, `predict_membership_proba` the membership score.

### 6.6 Evaluation (`src/evaluate.py`, `run_analysis.py`)
`evaluate_attack` returns overall and per-class precision, recall, accuracy and the median
per-class precision (the statistic of the paper's Fig. 4). `run_analysis.py` adds the
ROC curve, AUC, TPR at 1% and 0.1% FPR (highest TPR whose FPR does not exceed the limit),
a no-shadow baseline that thresholds the target's confidence in the true class, histograms
of that confidence and of the prediction entropy, the shadow-count sweep and the per-class
gap-vs-precision correlation.

### 6.7 Defenses (`src/mitigations.py`, `src/dp.py`)
* `top_k`: keep the k largest probabilities, zero the rest.
* `top1_label_only`: one-hot vector of the predicted class.
* `rounding`: round to d decimals.
* `temperature`: softmax(logits / t), computed from the probabilities (identical result).
* L2 and dropout: training options of `train_model` / `CifarCNN`.
* `train_model_dp`: DP-SGD with Opacus (`make_private_with_epsilon`): per-example gradient
  clipping (norm 1.0), Gaussian noise calibrated so the whole run is
  (epsilon, delta = 1e-5)-differentially private, Poisson sampling, batch size 128,
  20 epochs, Adam lr 0.001.

### 6.8 Experiment scripts
| Script | Saves under | Content |
|---|---|---|
| `python -m src.evaluate` | (prints) | Full pipeline test, builds the shadow cache |
| `run_experiments.py` | `sweep`, `mitigation`, `l2` | Size sweep, output defenses, L2 sweep |
| `run_analysis.py` | `analysis` | ROC/AUC, histograms, shadow sweep, gap vs precision |
| `run_defense_experiments.py` | `dropout` | Dropout and dropout + L2 |
| `run_dp_experiments.py` | `dp_sgd` | DP-SGD at epsilon 1, 5, 10, 50 |
| `run_seeds.py` | `repeats` | Baseline and output defenses over 5 seeds |
| `build_demo.py` | `results/demo/` | Models for the live demo |
| `run_all.py` | all of the above | Runs everything in order |

All results go to `results/results.json`, which the dashboard reads.

---

## 7. Experimental setup

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 (60,000 32x32 colour images, 10 classes) |
| Target / shadow train size | 5,000 (sweep: 2,500 / 5,000 / 10,000 / 15,000) |
| Members / non-members evaluated | 5,000 / 5,000 |
| Target and shadow training | Adam, lr 0.001, batch 64, 20 epochs, Tanh CNN |
| Shadow models | 10 (main run, analysis, seed repeats), 3 (size, L2, dropout, DP sweeps) |
| Attack models | 10 (one per class), FC 64 + ReLU, 50 epochs, Adam lr 0.001 |
| Membership threshold | 0.5 on the attack model's softmax |
| Seeds | Data split 42, target 42 + 999, shadow i 42 + i, attack class c 42 + c; repeats use seeds 42, 1, 2, 3, 4 |
| DP-SGD | epsilon in {1, 5, 10, 50}, delta 1e-5, clip norm 1.0, batch 128, 20 epochs |
| Hardware used for the reported run | NVIDIA RTX 3050 Laptop GPU (CNNs), CPU (attack models) |
| Software used for the reported run | Python 3.12, torch 2.6.0 (CUDA 12.4), opacus 1.6.0 |

---

## 8. Results

All numbers below come from one clean run of the final code (Section 9). Unless stated
otherwise: CIFAR-10, target trained on 5,000 images, evaluated on its 5,000 members and
5,000 non-members (random guessing = 0.5), 10 shadow models.

### 8.1 Baseline attack (no defense)

| Metric | Value |
|---|---|
| Target train accuracy | 1.000 |
| Target test accuracy | 0.565 |
| Train-test gap | 0.435 |
| Attack precision | **0.790** |
| Attack recall | **0.984** |
| Attack accuracy | 0.861 |
| Median per-class precision (paper's Fig. 4 statistic) | 0.762 (paper: 0.74) |
| Attack AUC | 0.901 |
| AUC of the no-shadow confidence baseline | 0.856 |
| TPR at 1% FPR: shadow attack / baseline | 0.114 / 0.015 |
| TPR at 0.1% FPR: shadow attack / baseline | 0.009 / 0.001 |

**Run-to-run variation.** The baseline and all output defenses were repeated with 5 seeds
(42, 1, 2, 3, 4). Each seed draws a new target/shadow split and trains a new target, 10 new
shadow models and new attack models. Seed 42 is the main run above and reproduces it
exactly.

| Metric (no defense) | Mean ± std over 5 seeds |
|---|---|
| Target test accuracy | 0.559 ± 0.007 |
| Train-test gap | 0.441 ± 0.007 |
| Attack precision | 0.792 ± 0.007 |
| Attack recall | 0.976 ± 0.006 |
| Attack accuracy | 0.860 ± 0.005 |
| Median per-class precision | 0.775 ± 0.011 |
| Attack AUC | 0.900 ± 0.005 |
| Confidence-baseline AUC | 0.855 ± 0.008 |
| TPR at 1% FPR | 0.117 ± 0.014 |

| Output defense | Precision | Recall | Attack accuracy |
|---|---|---|---|
| None | 0.792 ± 0.007 | 0.976 ± 0.006 | 0.860 ± 0.005 |
| Top k = 3 | 0.793 ± 0.007 | 0.976 ± 0.006 | 0.861 ± 0.005 |
| Top k = 1 | 0.785 ± 0.008 | 0.979 ± 0.005 | 0.855 ± 0.007 |
| Top k = 1, label only | 0.641 ± 0.003 | 1.000 ± 0.000 | 0.721 ± 0.004 |
| Rounding d = 3 | 0.792 ± 0.007 | 0.976 ± 0.005 | 0.860 ± 0.005 |
| Rounding d = 1 | 0.779 ± 0.009 | 0.986 ± 0.003 | 0.853 ± 0.007 |
| Temperature t = 5 | 0.785 ± 0.007 | 0.963 ± 0.006 | 0.850 ± 0.005 |
| Temperature t = 20 | 0.774 ± 0.006 | 0.912 ± 0.018 | 0.823 ± 0.007 |

With a standard deviation of about 0.007, precision differences below about 0.015 are
noise: top-k, rounding d = 3 and temperature t = 5 are statistically indistinguishable from
no defense, and the AUC gap between the shadow attack and the confidence baseline (0.045)
is real.

**Interpretation.** The target memorizes its training set (100% train accuracy, 57% test
accuracy). The attack exploits exactly this: of the images it calls members, 79% are
members, and it finds 98% of all members. The result is the same order as the paper's
CIFAR-10 result (0.74 median precision at 5,000 images, recall almost 1).

### 8.2 Why the attack works

**Members and non-members look different.** 97.7% of members get a confidence of at least
0.95 in their true class, against only 26.6% of non-members; 21.1% of non-members get less
than 0.05 (the model is confidently wrong). The dashboard's "Why the attack works" page
shows the full histograms of this confidence and of the prediction entropy (the analogue
of the paper's Fig. 12).

**Overfitting per class predicts leakage** (paper, Fig. 11). Correlation between a class's
train-test gap and the attack precision on that class: **0.96**.

| Class | Train acc | Test acc | Gap | Attack precision | Attack recall |
|---|---|---|---|---|---|
| airplane | 1.000 | 0.612 | 0.388 | 0.755 | 0.992 |
| automobile | 1.000 | 0.667 | 0.333 | 0.748 | 0.982 |
| bird | 1.000 | 0.400 | 0.600 | 0.867 | 0.996 |
| cat | 1.000 | 0.348 | 0.652 | **0.903** | 0.985 |
| deer | 1.000 | 0.521 | 0.479 | 0.829 | 0.989 |
| dog | 1.000 | 0.499 | 0.501 | 0.865 | 0.984 |
| frog | 1.000 | 0.690 | 0.310 | **0.705** | 0.983 |
| horse | 1.000 | 0.642 | 0.358 | 0.741 | 0.986 |
| ship | 1.000 | 0.686 | 0.314 | 0.752 | 0.964 |
| truck | 1.000 | 0.615 | 0.385 | 0.769 | 0.976 |

The hardest classes for the model (cat, bird, dog) are the most exposed: the model can only
get them right by memorizing, so its behaviour on members and non-members differs most.

**Shadow attack vs a simple threshold.** Thresholding the target's confidence in the true
class, with no shadow models at all (Yeom et al. 2018), already reaches AUC 0.856. The
shadow-model attack improves this to 0.901, and the gain is largest where it matters most,
at low false-positive rates (TPR at 1% FPR 0.114 vs 0.015). Even so, at a 0.1% false-positive
rate it identifies fewer than 1% of members: the attack is strong on average but weak at
confidently exposing *specific* individuals.

### 8.3 Effect of the training-set size (paper, Fig. 4)

3 shadow models per size.

| Train size | Target train acc | Target test acc | Gap | Attack precision | Attack recall | Attack accuracy | Median class precision | Paper (Fig. 4, median) |
|---|---|---|---|---|---|---|---|---|
| 2,500 | 1.000 | 0.502 | 0.498 | 0.817 | 0.962 | 0.874 | **0.814** | 0.78 |
| 5,000 | 1.000 | 0.565 | 0.435 | 0.786 | 0.985 | 0.859 | **0.758** | 0.74 |
| 10,000 | 1.000 | 0.595 | 0.405 | 0.765 | 0.984 | 0.841 | **0.730** | 0.72 |
| 15,000 | 1.000 | 0.624 | 0.376 | 0.747 | 0.983 | 0.825 | **0.718** | 0.71 |

Our median per-class precision is within 0.04 of the paper at every size, and follows the
same monotone decrease: more training data, less overfitting, less leakage. Recall stays
close to 1, as in the paper.

### 8.4 Number of shadow models

The attack models were retrained on the first k of the 10 cached shadow models.

| Shadow models | Precision | Recall | Accuracy | AUC |
|---|---|---|---|---|
| 1 | 0.766 | 0.992 | 0.845 | 0.893 |
| 2 | 0.780 | 0.988 | 0.855 | 0.896 |
| 5 | 0.791 | 0.981 | 0.861 | 0.897 |
| 10 | 0.790 | 0.984 | 0.861 | 0.901 |

A single shadow model already gives most of the attack's power; there is little gain beyond
5. Using 10 (instead of the paper's 100) is therefore not a meaningful handicap.

### 8.5 Defenses

In every row the attacker knows the defense: shadow outputs get the same output defense,
shadow models are trained with the same training defense. L2, dropout and DP-SGD rows use
3 shadow models.

| Defense | Target train acc | Target test acc | Attack accuracy | Attack precision | Attack recall |
|---|---|---|---|---|---|
| None | 1.000 | 0.565 | 0.861 | 0.790 | 0.984 |
| Top k = 3 | 1.000 | 0.565 | 0.862 | 0.791 | 0.983 |
| Top k = 1 | 1.000 | 0.565 | 0.854 | 0.781 | 0.984 |
| Top k = 1, label only | 1.000 | 0.565 | 0.717 | 0.639 | 1.000 |
| Rounding d = 3 | 1.000 | 0.565 | 0.861 | 0.790 | 0.983 |
| Rounding d = 1 | 1.000 | 0.565 | 0.849 | 0.773 | 0.988 |
| Temperature t = 5 | 1.000 | 0.565 | 0.850 | 0.781 | 0.973 |
| Temperature t = 20 | 1.000 | 0.565 | 0.821 | 0.771 | 0.915 |
| L2, lambda = 1e-4 | 1.000 | 0.565 | 0.856 | 0.783 | 0.984 |
| L2, lambda = 1e-3 | 0.996 | 0.542 | 0.793 | 0.738 | 0.909 |
| L2, lambda = 3e-3 | 0.952 | 0.525 | 0.738 | 0.691 | 0.860 |
| L2, lambda = 5e-3 | 0.871 | 0.538 | 0.674 | 0.637 | 0.812 |
| L2, lambda = 1e-2 | 0.673 | 0.547 | 0.565 | **0.552** | 0.687 |
| Dropout 0.2 | 1.000 | 0.557 | 0.826 | 0.762 | 0.947 |
| Dropout 0.5 | 0.987 | 0.550 | 0.770 | 0.708 | 0.920 |
| Dropout 0.5 + L2 1e-3 | 0.957 | 0.539 | 0.732 | 0.683 | 0.866 |
| DP-SGD, epsilon = 50 | 0.495 | 0.458 | 0.519 | 0.515 | 0.638 |
| DP-SGD, epsilon = 10 | 0.438 | 0.415 | 0.513 | 0.510 | 0.624 |
| DP-SGD, epsilon = 5 | 0.403 | 0.388 | 0.514 | 0.512 | 0.590 |
| DP-SGD, epsilon = 1 | 0.324 | 0.321 | 0.511 | **0.510** | 0.587 |

The privacy budget actually spent by Opacus was 49.99, 10.00, 5.00 and 1.00 (delta 1e-5).

**Output defenses (paper, Table III).** Top-k, rounding and temperature leave the attack
almost untouched (precision 0.77 to 0.79), exactly as the paper found on its datasets: the
largest probability alone still reveals how sure the model is. Releasing **only the label**
is the one output defense that helps (precision 0.639), but it does not stop the attack. The
attack learns to call an image a member whenever the model classifies it correctly. Since
the model is right on 100% of members but only 56.5% of non-members, this gives precision
1 / (1 + 0.565) = 0.639 and recall 1.000, which is exactly what we measure.

**L2 regularization** helps in proportion to its strength. With the paper's definition of
lambda, 1e-2 brings precision down to 0.552 (attack accuracy 0.565), closest to chance of
all non-DP defenses, while test accuracy drops only from 0.565 to 0.547. The paper reports
the same pattern on the Purchase dataset (L2 1e-2: precision 0.54) and warns that
stronger regularization costs accuracy. In our setting the model's train accuracy falls to
0.673 at lambda = 1e-2, i.e. it no longer fits its training set.

**Dropout** helps less than strong L2 at similar accuracy (precision 0.708 at p = 0.5).

**DP-SGD** brings the attack to chance at every tested epsilon (precision 0.510 to 0.515,
accuracy 0.511 to 0.519): the train-test gap all but disappears (0.003 to 0.037). The price
is utility: test accuracy falls by 11 to 24 points (0.458 at epsilon = 50, 0.321 at
epsilon = 1). The model learns too little, under noise and clipping, to memorize individual
images.

**Privacy vs utility.** No defense here keeps full accuracy and removes the leak. Strong L2
gives the best trade-off in this setup (attack accuracy 0.565 at 1.8 points of test accuracy
lost); DP-SGD is the only one with a formal guarantee and fully neutralizes this attack, at
a large accuracy cost with untuned settings.

---

## 9. Verification and audit

The project was checked end to end before these final results were produced.

### 9.1 What was verified

1. **The originally committed results are genuine.** The saved demo models in the
   repository were re-loaded and re-evaluated on a fresh copy of CIFAR-10 (original archive,
   md5 `c58f30108f718f92721af3b95e74349a`). Every committed number was reproduced exactly:
   target train/test accuracy of all four saved targets, and attack precision/recall of all
   five saved attacks (e.g. precision 0.7952, recall 0.9762 without defense; 0.6412 / 1.000
   label-only; 0.5121 / 0.5272 DP-SGD).
2. **Retraining is close but not bit-identical across machines.** Retraining the baseline
   target from scratch with the original code on a different CPU gave test accuracy 0.558
   instead of 0.5596. Floating-point summation order differs between hardware and library
   builds, which is why the repeated-seed experiment was added.
3. **The method matches the paper** (Section 3.4): disjoint target/shadow data, equal
   member/non-member counts, per-class attack models with the paper's architecture, the
   paper's CNN, and shadows that copy any defense.
4. **Internal consistency checks** on the final results: the 10-shadow entry of the
   shadow-count sweep equals the main result; the L2 lambda = 0 entry equals the size-sweep
   5,000 entry (same target, same 3 shadows, same attack seeds); the epsilon actually spent
   by Opacus matches every requested epsilon; the live-demo models (built by a separate
   script) give exactly the experiment numbers for no defense, label-only, L2 1e-2 and
   dropout 0.5; seed 42 of the repeated runs reproduces the main run exactly, so GPU
   training is bit-for-bit repeatable on the same machine.

### 9.2 Problems found and fixed

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | **L2 strength was half the paper's lambda.** L2 used Adam's `weight_decay=lambda`, which adds `lambda * theta` to the gradient, i.e. a penalty of `lambda/2 * sum(theta^2)`. The paper adds `lambda * sum(theta^2)` to the loss. | Every reported L2 setting was really half as strong as labeled. Confirmed by the rerun: the corrected lambda = 5e-3 gives precision 0.637, the old "lambda = 1e-2" gave 0.634. | Penalty added to the loss exactly as in the paper (`src/train.py`, argument renamed `l2`). L2 grid now 1e-4, 1e-3, 3e-3, 5e-3, 1e-2. |
| 2 | **TPR at low FPR was computed with `np.interp` on the ROC curve.** The ROC curve has repeated FPR values, where interpolation is not well defined, and interpolation is not the standard metric. | TPR at 1% / 0.1% FPR could be slightly off. | Standard definition: highest TPR among thresholds with FPR at or below the limit (`run_analysis.py`). |
| 3 | **Attack models were not seeded.** Their initialization depended on everything run earlier in the same script. | Results changed with the order of experiments; identical settings could give different numbers. | Each class's attack model is seeded with `SEED + class` (`src/attack.py`). |
| 4 | **Stale shadow cache could be reused silently.** The cache had no record of the settings it was built with. | Changing `config.py` (e.g. number of shadows) would still load old shadow data. | Settings are stored with the cache and checked on load; mismatches retrain (`src/shadow.py`). |
| 5 | **`src.evaluate` trained a differently seeded target** from the other scripts. | Step 1 printed numbers that did not match the dashboard. | Same target seed everywhere (`config.TARGET_SEED_OFFSET`). |
| 6 | **Single run only.** No measure of run-to-run noise. | Small differences between defenses could not be judged. | `run_seeds.py`: 5 seeds, mean ± std, shown in the dashboard. |
| 7 | **Paper comparison used a different statistic.** The paper's Fig. 4 reports *median per-class* precision; only overall precision was reported. | Comparison with the paper was approximate. | Median per-class precision is computed and compared directly. |
| 8 | Seeding did not cover Python's `random`, the GPU, or cuDNN. | GPU runs not repeatable. | `set_seed` extended. |
| 9 | Dashboard axis ranges were hard-coded. | New results could fall outside the plotted range. | Ranges computed from the data; paper Fig. 4 values and seed statistics added. |
| 10 | Slow training loops: `DataLoader` fetched and stacked samples one at a time, and the tiny attack MLP ran on the GPU. | Full pipeline took 6+ hours on CPU. | Tensor slicing for mini-batches, attack models on CPU, results saved per key so scripts can run in parallel. |
| 11 | **Wrong explanation of the label-only result** in the dashboard ("members and non-members are misclassified differently", the paper's argument for its datasets). | Misleading interpretation. | Our attack simply calls correctly classified images members: precision = 1 / (1 + test accuracy) = 0.639 exactly. Caption and report corrected. |
| 12 | `load_cifar` made several full float copies of the dataset (~700 MB each). | Out-of-memory errors on 8 GB laptops when running scripts side by side. | Scaling and standardization done in place; the output is bit-identical (verified by md5). |
| 13 | Leftover files `results/demo/{target.pt, attack_models.pt, info.json}` from an earlier demo version; unused `matplotlib` / `tqdm` requirements. | Clutter. | Removed. |

All final numbers in Section 8 come from a single clean run of the corrected code.

---

## 10. Discussion

### 10.1 Did we reproduce the paper?

Yes, for everything in scope:

| Paper claim | Our result |
|---|---|
| CIFAR-10 CNNs leak membership; median precision 0.78 / 0.74 / 0.72 / 0.71 for 2,500 to 15,000 training images, recall almost 1 (Fig. 4) | 0.81 / 0.76 / 0.73 / 0.72, recall 0.96 to 0.99 |
| More training data reduces leakage | Precision falls monotonically with size |
| Leakage tracks overfitting per class (Fig. 11) | Correlation 0.96 between per-class gap and precision |
| Top-k, rounding, temperature barely help (Table III) | Precision 0.77 to 0.79 vs 0.79 undefended |
| Even the label alone leaks (Table III) | Precision 0.639 with label-only output |
| L2 regularization helps; strong L2 nearly removes the leak (Table III) | Precision 0.552 at lambda = 1e-2 |

The paper's defense experiments were on Purchase and Texas hospital data; we observe the
same pattern on CIFAR-10.

### 10.2 What the results mean

* **Overfitting is the root cause.** Every result points the same way: the more a model's
  behaviour on its training data differs from its behaviour on new data, the more it leaks.
  Defenses that reduce this difference (L2, DP-SGD) work; defenses that only hide parts of
  the output (top-k, rounding, temperature) do not, because the remaining output still
  reflects the difference.
* **Accuracy is not privacy.** The undefended and the L2 = 1e-2 models have almost the same
  test accuracy (0.565 vs 0.547) but very different leakage (attack accuracy 0.861 vs 0.565).
  Train-test gap, not test accuracy, is what an auditor should look at.
* **Average-case vs worst-case risk.** Precision 0.79 sounds alarming, but the attack can
  only expose about 11% of members while keeping false accusations at 1%, and under 1% at
  0.1%. Whether that is acceptable depends on the application; for sensitive data (medical,
  financial) even a few confidently exposed records are a breach.
* **Shadow models are less essential than they look.** A simple confidence threshold
  (no shadows) reaches AUC 0.856 vs 0.901, and one shadow model is almost as good as ten.
  The shadow technique's main value is calibration: it tells the attacker where to set the
  threshold, which matters most at low false-positive rates.
* **Differential privacy works but costs utility.** DP-SGD is the only defense with a
  provable bound, and the attack is at chance against it, but with untuned settings the
  model loses 11 to 24 points of accuracy.

### 10.3 Practical recommendations

1. Measure the train-test gap before releasing a model; treat a large gap as a privacy
   risk, not only as a generalization problem.
2. Use regularization (L2, early stopping, data augmentation) as a first, almost free
   step.
3. For sensitive data, train with DP-SGD and tune it (clipping norm, batch size, epochs)
   to recover accuracy.
4. Do not rely on output restrictions (top-k, rounding) as a privacy measure.
5. Report membership-inference risk with TPR at low FPR, not only average precision or
   accuracy.

---

## 11. Limitations

* **CIFAR-10 only.** The paper's other datasets, CIFAR-100 and the commercial ML services
  are not covered, so conclusions about those settings come from the paper, not from us.
* **Fewer shadow models and epochs than the paper** (10 or 3 shadows instead of 100; 20
  epochs with Adam instead of up to 100). The shadow-count sweep shows little gain beyond
  about 5 shadows, so this is unlikely to change the conclusions.
* **Seed variance was measured only for the baseline and output defenses.** The
  training-defense sweeps (L2, dropout, DP-SGD) are single runs; use the baseline standard
  deviations as a guide to their noise.
* **The defenses were evaluated against this attack only.** Stronger attacks (e.g. LiRA,
  Carlini et al. 2022) could do better, especially at low false-positive rates. A failed
  attack against DP-SGD models is an empirical observation; the formal guarantee comes from
  the DP accounting, not from our experiment.
* **DP-SGD was not tuned** (clipping norm, batch size, learning rate, epochs). Tuned DP
  training would keep more accuracy at the same epsilon. Part of the DP models' privacy
  comes simply from their lower capacity to fit the training set.
* **Precision depends on the 50/50 member ratio.** In practice members are usually rare, and
  then precision is much lower than reported here; TPR at low FPR is the more realistic
  measure.

---

## 12. How to reproduce

### 12.1 Setup

```bash
git clone https://github.com/DIRAVINA/cifar-mia.git
cd cifar-mia
python -m venv venv
venv\Scripts\activate            # Windows PowerShell
# source venv/bin/activate       # Mac / Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # only for an NVIDIA GPU
pip install -r requirements.txt
```

CIFAR-10 (about 170 MB) downloads automatically to `data/` on the first run.

### 12.2 View the results

```bash
streamlit run app.py
```

The saved `results/results.json` and `results/demo/` are in the repository, so nothing has
to be retrained to explore the dashboard and live demo.

### 12.3 Re-run everything

```bash
python run_all.py
```

| Step | Command | Content | Time (laptop GPU) |
|---|---|---|---|
| 1 | `python -m src.evaluate` | Target, 10 shadows, attack; builds the shadow cache | ~3 min |
| 2 | `python run_experiments.py --exp all` | Size sweep, output defenses, L2 | ~40 min |
| 3 | `python run_analysis.py` | ROC/AUC, histograms, shadow sweep, gap vs precision | ~10 min |
| 4 | `python run_defense_experiments.py` | Dropout | ~10 min |
| 5 | `python run_dp_experiments.py` | DP-SGD | ~20 min |
| 6 | `python run_seeds.py` | 5-seed repeats | ~50 min |
| 7 | `python build_demo.py --force` | Live demo models | ~15 min |

Steps 2 to 7 only need step 1 to have run, so they can also be started in parallel in
separate terminals (results are written to `results.json` under a lock, so they do not
overwrite each other). Each process needs about 2 GB of RAM and some GPU memory: on an
8 GB laptop with a 4 GB GPU, run at most 3 at a time. `run_seeds.py --seeds 3 4` runs a
subset of the seeds; each finished seed is merged into the saved results.

Expect results within a few thousandths of the reported ones on other hardware (Section
9.1).

### 12.4 Quick checks

```bash
python -m src.data      # data split shapes
python -m src.model     # model output shape and parameter count
python -m src.train     # trains one target, prints the train-test gap
python -m src.dp        # 3-epoch DP-SGD smoke test
```

---

## 13. Code reference

```
cifar-mia/
├── config.py                    all settings (sizes, epochs, seeds, defenses)
├── run_all.py                   runs the whole pipeline in order
├── run_experiments.py           size sweep, output defenses, L2
├── run_analysis.py              ROC/AUC, histograms, shadow sweep, gap vs precision
├── run_defense_experiments.py   dropout, dropout + L2
├── run_dp_experiments.py        DP-SGD
├── run_seeds.py                 repeated runs over 5 seeds
├── build_demo.py                models for the live demo
├── app.py                       Streamlit dashboard
├── pages/1_Live_Demo.py         live demo page
├── src/
│   ├── data.py                  loading, seeding, target/shadow split
│   ├── model.py                 CifarCNN
│   ├── train.py                 training (with paper L2), prediction, accuracy
│   ├── shadow.py                shadow models, attack data, cache
│   ├── attack.py                per-class attack models
│   ├── evaluate.py              metrics, full pipeline test
│   ├── mitigations.py           top-k, label only, rounding, temperature
│   └── dp.py                    DP-SGD training with Opacus
├── results/
│   ├── results.json             all experiment results
│   └── demo/<defense>/          target + attack models per demo defense
└── docs/PROJECT_REPORT.md       this report
```

### Dashboard sections
* **Overview**: key numbers, seed statistics, attack pipeline diagram.
* **Why the attack works**: confidence and entropy histograms, ROC curves vs the
  confidence baseline, per-class gap vs precision.
* **Attack strength**: training-set size (with the paper's Fig. 4 values), number of
  shadow models.
* **Defenses**: output defenses (with seed statistics), L2, dropout, DP-SGD,
  privacy-vs-utility scatter.
* **Per-class results**: heatmap over all experiments and a per-class drill-down.
* **Live Demo** page: random members and non-members, the attack's verdict for each, a
  defense selector that re-scores the same images, and a batch test.

---

## 14. References

1. R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against
   Machine Learning Models.* IEEE Symposium on Security and Privacy, 2017.
2. M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. *Deep
   Learning with Differential Privacy.* ACM CCS, 2016.
3. A. Yousefpour et al. *Opacus: User-Friendly Differential Privacy Library in PyTorch.*
   2021.
4. S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. *Privacy Risk in Machine Learning:
   Analyzing the Connection to Overfitting.* IEEE CSF, 2018. (Confidence-threshold
   baseline.)
5. N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr. *Membership Inference
   Attacks From First Principles.* IEEE S&P, 2022. (TPR at low FPR as the evaluation
   standard.)
6. A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* 2009.
   (CIFAR-10.)

---

## 15. Appendix: glossary

| Term | Meaning |
|---|---|
| Member / non-member | A record that was / was not in the target model's training set |
| Target model | The victim model the attacker queries |
| Shadow model | A model the attacker trains to imitate the target, with known membership |
| Attack model | Classifier from prediction vector to member / non-member |
| Prediction vector | The model's output probabilities, one per class |
| Train-test gap | Training accuracy minus test accuracy; a measure of overfitting |
| Precision / recall | See Section 2.3 |
| ROC / AUC | True-positive rate vs false-positive rate over all thresholds / area under it |
| TPR at x% FPR | Members found while falsely flagging at most x% of non-members |
| Differential privacy (epsilon, delta) | A guarantee that one record changes the output distribution of training by at most a factor e^epsilon (plus delta); smaller epsilon = more private |
| DP-SGD | SGD with per-example gradient clipping and Gaussian noise, giving differential privacy |
