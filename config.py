# config.py
# All experiment settings live here. Change numbers here, not inside other files.

SEED = 42

# --- Dataset ---
DATASET = "cifar10"          # "cifar10" or "cifar100"
DATA_DIR = "./data"          # CIFAR is downloaded here automatically

# --- Target model ---
TARGET_TRAIN_SIZE = 5000     # paper tries 2500 / 5000 / 10000 / 15000 for CIFAR-10
TARGET_EPOCHS = 20         # paper uses up to 100 (we want it to overfit)

# --- Shadow models ---
NUM_SHADOW = 10              # paper uses 100; start small so it runs on a laptop
SHADOW_TRAIN_SIZE = 5000     # same size as target train, as in the paper
SHADOW_EPOCHS = 20

# --- Training hyperparameters (paper: lr 0.001, Tanh activation) ---
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
L2_LAMBDAS = [0.0, 1e-4, 1e-3]                # 0.0 = no regularization (baseline)