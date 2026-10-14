# src/shadow.py
import json
import os
import numpy as np
import torch

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN
from src.train import train_model, predict_proba, accuracy


def build_attack_data(model, X_in, y_in, X_out, y_out, device):
    """
    Queries a model with its member images (X_in) and non-member images (X_out).
    Returns the raw material for the attack model:

        probs     : prediction vectors, shape (N, num_classes)
        labels    : TRUE class label of each image, shape (N,)
        is_member : 1 = "in" (was in training set), 0 = "out", shape (N,)

    The same function is used for shadow models (training the attack) and,
    later, for the target model (evaluating the attack).
    """
    probs_in = predict_proba(model, X_in, device)
    probs_out = predict_proba(model, X_out, device)

    probs = np.concatenate([probs_in, probs_out], axis=0)
    labels = np.concatenate([y_in, y_out], axis=0)
    is_member = np.concatenate([np.ones(len(X_in)), np.zeros(len(X_out))]).astype(np.int64)
    return probs, labels, is_member


def train_shadow_models(X_pool, y_pool, num_classes,
                        num_shadow=config.NUM_SHADOW,
                        train_size=config.SHADOW_TRAIN_SIZE,
                        epochs=config.SHADOW_EPOCHS,
                        device=None, seed=config.SEED, l2=0.0,
                        dropout=0.0):
    """
    Trains `num_shadow` shadow models. For each shadow i:
      1. randomly pick `train_size` images from the pool as its training set
      2. pick another `train_size` images (disjoint from #1) as its test set
      3. train a fresh CifarCNN on the training set
      4. query it on both sets and label the outputs in / out

    Different shadows may reuse the same images (allowed by the paper),
    but the pool never contains target-model data.

    l2 is the L2 regularization strength, paper lambda (0.0 = none).
    dropout is the dropout probability (0.0 = none). The attacker is assumed
    to know the defense, so shadows are trained with the same settings.

    Returns the combined attack training data from all shadows.
    """
    rng = np.random.RandomState(seed)
    all_probs, all_labels, all_member = [], [], []

    for i in range(num_shadow):
        print(f"[Shadow {i + 1}/{num_shadow}] training...")
        idx = rng.permutation(len(X_pool))
        tr_idx = idx[:train_size]
        te_idx = idx[train_size: 2 * train_size]

        X_tr, y_tr = X_pool[tr_idx], y_pool[tr_idx]
        X_te, y_te = X_pool[te_idx], y_pool[te_idx]

        torch.manual_seed(seed + i)          # different init for each shadow
        model = CifarCNN(num_classes, dropout=dropout)
        train_model(model, X_tr, y_tr, epochs=epochs, device=device,
                    l2=l2, verbose=False)

        tr_acc = accuracy(model, X_tr, y_tr, device)
        te_acc = accuracy(model, X_te, y_te, device)
        print(f"    train acc {tr_acc:.3f} | test acc {te_acc:.3f} | gap {tr_acc - te_acc:.3f}")

        probs, labels, member = build_attack_data(model, X_tr, y_tr, X_te, y_te, device)
        all_probs.append(probs)
        all_labels.append(labels)
        all_member.append(member)

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "is_member": np.concatenate(all_member, axis=0),
    }


SHADOW_CACHE = os.path.join(config.RESULTS_DIR, "shadow_attack_data.npz")


def cache_settings():
    """Settings the cached shadow data depends on. A mismatch means it is stale."""
    return {"dataset": config.DATASET, "seed": config.SEED,
            "target_train_size": config.TARGET_TRAIN_SIZE,
            "num_shadow": config.NUM_SHADOW, "train_size": config.SHADOW_TRAIN_SIZE,
            "epochs": config.SHADOW_EPOCHS, "lr": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE}


def save_attack_data(data, path, settings=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    extra = {"settings": np.array(json.dumps(settings))} if settings else {}
    np.savez_compressed(path, **data, **extra)


def load_attack_data(path, expected_settings=None):
    """
    Loads cached attack data. If expected_settings is given and the cache was
    made with different settings, returns None (the caller should retrain).
    """
    f = np.load(path)
    if expected_settings is not None:
        saved = json.loads(str(f["settings"])) if "settings" in f.files else None
        if saved != expected_settings:
            return None
    return {k: f[k] for k in f.files if k != "settings"}


def load_or_train_shadow_data(s, num_classes, device, path=SHADOW_CACHE):
    """
    The main shadow data (config.NUM_SHADOW shadows, default settings), shared
    by src.evaluate, run_experiments, run_analysis and build_demo. Trained once
    and cached; retrained automatically if config.py changed since.
    """
    settings = cache_settings()
    if os.path.exists(path):
        data = load_attack_data(path, settings)
        if data is not None:
            print("Loaded cached shadow data from", path)
            return data
        print("Cached shadow data was made with other settings; retraining.")
    data = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"],
                               num_classes, device=device)
    save_attack_data(data, path, settings)
    return data


if __name__ == "__main__":
    # Self-test: run with  python -m src.shadow
    # Uses only 2 shadow models and 10 epochs so it finishes quickly.
    set_seed()
    device = get_device()
    X, y = load_cifar()
    s = split_target_and_shadow_pool(X, y)
    num_classes = len(np.unique(y))

    data = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"], num_classes,
                               num_shadow=2, epochs=10, device=device)

    print("\nAttack training data built from shadow models:")
    print("  probs    :", data["probs"].shape)
    print("  labels   :", data["labels"].shape)
    print("  is_member:", data["is_member"].shape,
          "| members:", int(data["is_member"].sum()),
          "| non-members:", int((1 - data["is_member"]).sum()))
    print("  first prediction vector:", np.round(data["probs"][0], 3))