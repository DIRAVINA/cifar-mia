# config.py
# All experiment settings live here. Change numbers here, not inside other files.

SEED = 42
TARGET_SEED_OFFSET = 999     # target model init uses SEED + 999 (shadows use SEED + i)

# --- Dataset ---
DATASET = "cifar10"          # "cifar10" or "cifar100"
DATA_DIR = "./data"          # CIFAR is downloaded here automatically

# --- Target model ---
TARGET_TRAIN_SIZE = 5000     # paper tries 2500 / 5000 / 10000 / 15000 for CIFAR-10
TARGET_EPOCHS = 20           # paper uses up to 100; 20 is already enough to overfit

# --- Shadow models ---
NUM_SHADOW = 10              # paper uses 100; start small so it runs on a laptop
SHADOW_TRAIN_SIZE = 5000     # same size as target train, as in the paper
SHADOW_EPOCHS = 20

# --- Training hyperparameters (paper: lr 0.001, Tanh activation; we use Adam) ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# --- Attack model ---
ATTACK_EPOCHS = 50
ATTACK_BATCH_SIZE = 64

# --- Output ---
RESULTS_DIR = "./results"

# --- Experiments (run_experiments.py) ---
NUM_SHADOW_SWEEP = 3                          # fewer shadows for the slow sweeps
SWEEP_SIZES = [2500, 5000, 10000, 15000]      # same sizes as the paper (CIFAR-10)
# L2 lambda as defined in the paper: loss + lambda * sum(theta^2). 0.0 = none.
L2_LAMBDAS = [0.0, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2]

# --- Repeated runs (run_seeds.py): the baseline attack with different seeds,
# each seed giving a new data split, new target and new shadow models ---
REPEAT_SEEDS = [42, 1, 2, 3, 4]
NUM_SHADOW_REPEAT = 10

# --- Dropout experiments (run_defense_experiments.py) ---
# (name shown in the dashboard, dropout probability, L2 strength)
DROPOUT_SETTINGS = [
    ("Dropout 0.2", 0.2, 0.0),
    ("Dropout 0.5", 0.5, 0.0),
    ("Dropout 0.5 + L2 1e-3", 0.5, 1e-3),
]


# --- DP-SGD (run_dp_experiments.py) ---
DP_EPSILONS = [1.0, 5.0, 10.0, 50.0]   # privacy budgets to test (smaller = more private)
DP_DELTA = 1e-5                        # standard choice, must be below 1/(training set size)
DP_MAX_GRAD_NORM = 1.0                 # per-example gradient clipping threshold
DP_LEARNING_RATE = 0.001
DP_BATCH_SIZE = 128
DP_EPOCHS = 20

# --- Live demo defenses (build_demo.py and pages/1_Live_Demo.py) ---
# key: (label shown in the page, dropout, L2 strength, output transform)
DEMO_DEFENSES = {
    "none":       ("None (unprotected)",     0.0, 0.0,  "none"),
    "label_only": ("Label-only output",      0.0, 0.0,  "label_only"),
    "l2":         ("L2 regularization 1e-2", 0.0, 1e-2, "none"),
    "dropout":    ("Dropout 0.5",            0.5, 0.0,  "none"),
    "dp":         ("DP-SGD (epsilon = 10)",  0.0, 0.0,  "none"),
}
DEMO_DP_EPSILON = 10.0