# build_demo.py
# Trains the target model + attack models once and saves them for the dashboard.
# Run with:  python build_demo.py
import json
import os

import numpy as np
import torch

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN
from src.train import train_model, accuracy
from src.shadow import load_attack_data
from src.attack import train_attack_models

DEMO_DIR = os.path.join(config.RESULTS_DIR, "demo")
SHADOW_CACHE = os.path.join(config.RESULTS_DIR, "shadow_attack_data.npz")

if __name__ == "__main__":
    if not os.path.exists(SHADOW_CACHE):
        raise SystemExit("Shadow data not found. Run first:  python -m src.evaluate")

    os.makedirs(DEMO_DIR, exist_ok=True)
    set_seed()
    device = get_device()
    X, y = load_cifar()
    s = split_target_and_shadow_pool(X, y)
    num_classes = len(np.unique(y))

    # 1) Target model (the victim)
    print("Training target model...")
    torch.manual_seed(config.SEED + 999)
    target = CifarCNN(num_classes)
    train_model(target, s["X_target_train"], s["y_target_train"],
                epochs=config.TARGET_EPOCHS, device=device, verbose=False)
    tr = accuracy(target, s["X_target_train"], s["y_target_train"], device)
    te = accuracy(target, s["X_target_test"], s["y_target_test"], device)
    print(f"target: train acc {tr:.3f} | test acc {te:.3f}")
    torch.save(target.cpu().state_dict(), os.path.join(DEMO_DIR, "target.pt"))

    # 2) Attack models (one per class), trained on the cached shadow data
    print("Training attack models...")
    shadow_data = load_attack_data(SHADOW_CACHE)
    attack_models = train_attack_models(shadow_data, num_classes, device)
    state = {c: m.cpu().state_dict() for c, m in attack_models.items()}
    torch.save(state, os.path.join(DEMO_DIR, "attack_models.pt"))

    # 3) Small info file for the dashboard
    info = {"num_classes": num_classes, "dataset": config.DATASET,
            "train_acc": tr, "test_acc": te,
            "n_members": len(s["X_target_train"]),
            "n_nonmembers": len(s["X_target_test"])}
    with open(os.path.join(DEMO_DIR, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Saved demo models to", DEMO_DIR)