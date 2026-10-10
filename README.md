# Membership Inference Attack on CIFAR-10

Implementation of **Shokri et al., "Membership Inference Attacks Against Machine Learning Models"** (IEEE S&P 2017, [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)) on CIFAR-10, using PyTorch, with a Streamlit dashboard for the results.

The attack takes only black-box access to a trained model and decides whether a given image was in its training set. It works by training **shadow models** that mimic the target, then training **one attack model per class** on the shadow models' outputs.

## Project structure

```
cifar-mia/
├── config.py              # all settings (sizes, epochs, seeds)
├── run_experiments.py     # runs experiments, saves results/results.json
├── app.py                 # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── data.py            # load CIFAR, split target / shadow data
│   ├── model.py           # CNN (2 conv + FC-128, Tanh)
│   ├── train.py           # training and prediction functions
│   ├── shadow.py          # shadow models, attack training data
│   ├── attack.py          # per-class attack models
│   ├── evaluate.py        # precision / recall, full pipeline test
│   └── mitigations.py     # top-k, rounding, temperature defenses
└── results/results.json   # saved experiment results (used by the app)
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

Then install the packages:

```bash
pip install -r requirements.txt
```

## How to run

Run all commands from the project root with the virtual environment active.

| Command | What it does | Approx. time (CPU) |
|---|---|---|
| `python -m src.data` | Downloads CIFAR-10 and checks the data splits | 1-2 min |
| `python -m src.model` | Checks the CNN output shape | seconds |
| `python -m src.train` | Trains a quick target model, shows the overfitting gap | 3-5 min |
| `python -m src.shadow` | Trains 2 shadow models as a test | 5-8 min |
| `python -m src.evaluate` | Full pipeline: target, 10 shadows, attack, precision / recall | 15-25 min first run |
| `python run_experiments.py --exp mitigation` | Output-side defenses | 10-15 min |
| `python run_experiments.py --exp l2` | L2 regularization | 15-25 min |
| `python run_experiments.py --exp sweep` | Training-set size sweep | 30-60 min |
| `python run_experiments.py --exp all` | All three experiments | 1-2 h |
| `streamlit run app.py` | Opens the dashboard at http://localhost:8501 | - |

Notes:
- The first `src.evaluate` run caches shadow outputs to `results/shadow_attack_data.npz` (not tracked by git); later runs reuse it.
- Results are saved to `results/results.json`, so the dashboard works without re-running anything.
- Settings (number of shadow models, epochs, sizes) are in `config.py`.

## Results

Setup: CIFAR-10, target trained on 5,000 images (50% members / 50% non-members in the evaluation, so random guessing = 0.5). Attack precision and recall are overall values.

### Effect of the target's training-set size

| Train size | Target train acc | Target test acc | Attack precision | Attack recall |
|---|---|---|---|---|
| 2,500 | 1.000 | 0.482 | 0.816 | 0.974 |
| 5,000 | 1.000 | 0.560 | 0.793 | 0.976 |
| 10,000 | 1.000 | 0.592 | 0.770 | 0.982 |
| 15,000 | 1.000 | 0.620 | 0.752 | 0.978 |

Larger training sets overfit less, so the attack weakens, the same trend as the paper.

### Output-side defenses

| Defense | Target test acc | Attack precision | Attack recall |
|---|---|---|---|
| No mitigation | 0.560 | 0.795 | 0.976 |
| Top k = 3 | 0.560 | 0.798 | 0.972 |
| Top k = 1 | 0.560 | 0.789 | 0.977 |
| Top k = 1, label only | 0.560 | 0.641 | 1.000 |
| Rounding d = 3 | 0.560 | 0.795 | 0.973 |
| Rounding d = 1 | 0.559 | 0.782 | 0.986 |
| Temperature t = 5 | 0.560 | 0.796 | 0.957 |
| Temperature t = 20 | 0.560 | 0.786 | 0.890 |

Only returning the label alone noticeably reduces the attack, and it still beats random guessing.

### L2 regularization

| Lambda | Target train acc | Target test acc | Attack precision | Attack recall |
|---|---|---|---|---|
| 0 (none) | 1.000 | 0.560 | 0.793 | 0.976 |
| 1e-4 | 1.000 | 0.555 | 0.796 | 0.977 |
| 1e-3 | 1.000 | 0.560 | 0.773 | 0.967 |

## Implementation notes

- Target and shadow models use the same CNN (2 conv + max-pool layers, FC-128, Tanh), as in the paper.
- Optimizer is Adam (the paper used SGD in Torch7) so models overfit within ~20 epochs on CPU.
- 10 shadow models are used for the main attack (the paper uses 100 for CIFAR); 3 for the sweeps.
- The attacker is assumed to know the defense, so shadow models receive the same output transformation.

## Reference

R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE Symposium on Security and Privacy, 2017.