# src/data.py
import numpy as np
import torch
from torchvision import datasets

import config


def get_device():
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=config.SEED):
    """Makes results repeatable."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cifar(name=config.DATASET, data_dir=config.DATA_DIR):
    """
    Downloads CIFAR and merges its train (50k) and test (10k) parts into
    one pool of 60k images, so we can make our own disjoint splits.

    Returns:
        X: float32 array, shape (60000, 3, 32, 32), standardized
        y: int array, shape (60000,), class labels
    """
    if name == "cifar10":
        ds_class = datasets.CIFAR10
    elif name == "cifar100":
        ds_class = datasets.CIFAR100
    else:
        raise ValueError("DATASET must be 'cifar10' or 'cifar100'")

    train_ds = ds_class(root=data_dir, train=True, download=True)
    test_ds = ds_class(root=data_dir, train=False, download=True)

    X = np.concatenate([train_ds.data, test_ds.data], axis=0)       # (N, 32, 32, 3) uint8
    y = np.concatenate([np.array(train_ds.targets), np.array(test_ds.targets)])

    # Scale pixels to [0, 1], then move channels first: (N, 3, 32, 32)
    X = X.astype(np.float32) / 255.0
    X = np.transpose(X, (0, 3, 1, 2))

    # Standardize each color channel (mean 0, std 1)
    mean = X.mean(axis=(0, 2, 3), keepdims=True)
    std = X.std(axis=(0, 2, 3), keepdims=True)
    X = (X - mean) / std

    return X.astype(np.float32), y.astype(np.int64)


def split_target_and_shadow_pool(X, y, target_train_size=config.TARGET_TRAIN_SIZE,
                                 seed=config.SEED):
    """
    Splits data into:
      - target train  (the "members" the attacker wants to detect)
      - target test   (same size, the "non-members" used for evaluation)
      - shadow pool   (everything else, disjoint from target data)

    Equal-size members / non-members => random guessing accuracy is 0.5,
    exactly as in the paper.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))

    t_train = idx[:target_train_size]
    t_test = idx[target_train_size: 2 * target_train_size]
    pool = idx[2 * target_train_size:]

    return {
        "X_target_train": X[t_train], "y_target_train": y[t_train],
        "X_target_test": X[t_test],   "y_target_test": y[t_test],
        "X_shadow_pool": X[pool],     "y_shadow_pool": y[pool],
    }


if __name__ == "__main__":
    # Quick self-test: run with  python -m src.data
    set_seed()
    print("Device:", get_device())
    X, y = load_cifar()
    print("Full data:", X.shape, y.shape, "num classes:", len(np.unique(y)))
    splits = split_target_and_shadow_pool(X, y)
    for k, v in splits.items():
        print(k, v.shape)