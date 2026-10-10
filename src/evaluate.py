# src/evaluate.py
import os
import numpy as np

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN
from src.train import train_model, accuracy
from src.shadow import (build_attack_data, train_shadow_models,
                        save_attack_data, load_attack_data)
from src.attack import train_attack_models, predict_membership


def precision_recall(true_member, pred_member):
    """
    precision = of the records the attacker called "member", how many really were
    recall    = of the real members, how many the attacker found
    accuracy  = fraction of all guesses that were right (0.5 = random guessing)
    """
    tp = int(((pred_member == 1) & (true_member == 1)).sum())
    fp = int(((pred_member == 1) & (true_member == 0)).sum())
    fn = int(((pred_member == 0) & (true_member == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = float((pred_member == true_member).mean())
    return precision, recall, acc


def evaluate_attack(attack_models, target_data, num_classes, device):
    """Overall and per-class precision/recall against the target model."""
    pred = predict_membership(attack_models, target_data["probs"],
                              target_data["labels"], device)
    true = target_data["is_member"]

    p, r, a = precision_recall(true, pred)
    per_class = []
    for c in range(num_classes):
        m = target_data["labels"] == c
        pc, rc, ac = precision_recall(true[m], pred[m])
        per_class.append({"class": c, "precision": pc, "recall": rc, "accuracy": ac})
    return {"precision": p, "recall": r, "accuracy": a, "per_class": per_class}


if __name__ == "__main__":
    # Full pipeline test: python -m src.evaluate
    set_seed()
    device = get_device()
    X, y = load_cifar()
    s = split_target_and_shadow_pool(X, y)
    num_classes = len(np.unique(y))

    # 1) Train the TARGET model (the victim)
    print("=== Training target model ===")
    target = CifarCNN(num_classes)
    train_model(target, s["X_target_train"], s["y_target_train"],
                epochs=config.TARGET_EPOCHS, device=device, verbose=False)
    tr = accuracy(target, s["X_target_train"], s["y_target_train"], device)
    te = accuracy(target, s["X_target_test"], s["y_target_test"], device)
    print(f"target: train acc {tr:.3f} | test acc {te:.3f} | gap {tr - te:.3f}")

    # 2) Shadow models (cached on disk so this runs only once)
    cache = os.path.join(config.RESULTS_DIR, "shadow_attack_data.npz")
    if os.path.exists(cache):
        print("\n=== Loading cached shadow data ===")
        shadow_data = load_attack_data(cache)
    else:
        print(f"\n=== Training {config.NUM_SHADOW} shadow models (slow, one time) ===")
        shadow_data = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"],
                                          num_classes, device=device)
        save_attack_data(shadow_data, cache)

    # 3) Train one attack model per class
    print("\n=== Training attack models ===")
    attack_models = train_attack_models(shadow_data, num_classes, device)

    # 4) Attack the target: members = its train set, non-members = its test set
    target_data = {}
    probs, labels, member = build_attack_data(
        target, s["X_target_train"], s["y_target_train"],
        s["X_target_test"], s["y_target_test"], device)
    target_data = {"probs": probs, "labels": labels, "is_member": member}

    res = evaluate_attack(attack_models, target_data, num_classes, device)
    print("\n=== RESULTS (random guessing accuracy = 0.5) ===")
    print(f"Overall precision: {res['precision']:.3f}")
    print(f"Overall recall   : {res['recall']:.3f}")
    print(f"Overall accuracy : {res['accuracy']:.3f}")
    print("\nPer class:")
    for row in res["per_class"]:
        print(f"  class {row['class']}: precision {row['precision']:.3f} "
              f"recall {row['recall']:.3f}")