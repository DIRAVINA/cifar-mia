# build_demo.py
# Builds the models for the Live Demo, one set per defense.
#   python build_demo.py                 builds every set that is missing
#   python build_demo.py --only dp       builds just one set
#   python build_demo.py --force         rebuilds everything
import argparse
import io
import json
import os
from contextlib import redirect_stdout

import numpy as np
import torch

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN
from src.train import train_model
from src.shadow import train_shadow_models, load_attack_data
from src.attack import train_attack_models
from src.evaluate import evaluate_attack
from src.mitigations import top1_label_only
from src.dp import train_model_dp
from run_experiments import make_target_data, split_acc
from run_dp_experiments import train_shadow_models_dp

DEMO_DIR = os.path.join(config.RESULTS_DIR, "demo")
SHADOW_CACHE = os.path.join(config.RESULTS_DIR, "shadow_attack_data.npz")


def transform_probs(kind, probs):
    """What the outside world gets to see from the model."""
    return top1_label_only(probs) if kind == "label_only" else probs


def cpu_state(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}


def fit_attack(shadow_data, num_classes, device):
    with redirect_stdout(io.StringIO()):          # hide the per-class prints
        return train_attack_models(shadow_data, num_classes, device)


def load_base_target(num_classes):
    path = os.path.join(DEMO_DIR, "none", "target.pt")
    if not os.path.exists(path):
        raise SystemExit("Build the 'none' set first:  python build_demo.py --only none")
    model = CifarCNN(num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def build_one(key, s, num_classes, device):
    label, dropout, l2, kind = config.DEMO_DEFENSES[key]
    out_dir = os.path.join(DEMO_DIR, key)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== {label} =====")

    # 1) the target model
    if key == "label_only":
        target = load_base_target(num_classes)           # same model, hidden outputs
    elif key == "dp":
        print("Training the DP-SGD target model...")
        target, _ = train_model_dp(s["X_target_train"], s["y_target_train"],
                                   num_classes, config.DEMO_DP_EPSILON, device,
                                   seed=config.SEED + 999)
    else:
        print("Training the target model...")
        torch.manual_seed(config.SEED + 999)
        target = CifarCNN(num_classes, dropout=dropout)
        train_model(target, s["X_target_train"], s["y_target_train"],
                    epochs=config.TARGET_EPOCHS, device=device,
                    weight_decay=l2, verbose=False)

    # 2) shadow data (the attacker knows the defense, so shadows get it too)
    if key in ("none", "label_only"):
        shadow = dict(load_attack_data(SHADOW_CACHE))
    elif key == "dp":
        shadow = train_shadow_models_dp(s["X_shadow_pool"], s["y_shadow_pool"],
                                        num_classes, config.DEMO_DP_EPSILON, device,
                                        num_shadow=config.NUM_SHADOW_SWEEP)
    else:
        shadow = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"], num_classes,
                                     num_shadow=config.NUM_SHADOW_SWEEP, device=device,
                                     weight_decay=l2, dropout=dropout)
    shadow["probs"] = transform_probs(kind, shadow["probs"])

    # 3) attack models + overall evaluation on the target
    print("Training attack models...")
    attack_models = fit_attack(shadow, num_classes, device)
    td = make_target_data(target, s, device)
    train_acc, test_acc = split_acc(td)
    td["probs"] = transform_probs(kind, td["probs"])
    res = evaluate_attack(attack_models, td, num_classes, device)
    print(f"  target train {train_acc:.3f} | test {test_acc:.3f} | attack precision "
          f"{res['precision']:.3f} | recall {res['recall']:.3f} | accuracy {res['accuracy']:.3f}")

    # 4) save
    if key != "label_only":
        torch.save(cpu_state(target), os.path.join(out_dir, "target.pt"))
    torch.save({c: cpu_state(m) for c, m in attack_models.items()},
               os.path.join(out_dir, "attack_models.pt"))
    info = {"key": key, "label": label, "dropout": dropout, "transform": kind,
            "target_dir": "none" if key == "label_only" else key,
            "num_classes": num_classes, "dataset": config.DATASET,
            "train_acc": train_acc, "test_acc": test_acc,
            "attack_precision": res["precision"], "attack_recall": res["recall"],
            "attack_accuracy": res["accuracy"]}
    if key == "dp":
        info["epsilon"] = config.DEMO_DP_EPSILON
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(config.DEMO_DEFENSES.keys()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(SHADOW_CACHE):
        raise SystemExit("Shadow data not found. Run first:  python -m src.evaluate")

    set_seed()
    device = get_device()
    X, y = load_cifar()
    s = split_target_and_shadow_pool(X, y)
    num_classes = len(np.unique(y))

    keys = [args.only] if args.only else list(config.DEMO_DEFENSES.keys())
    for key in keys:
        if os.path.exists(os.path.join(DEMO_DIR, key, "info.json")) and not args.force:
            print(f"\n[skip] '{key}' is already built (use --force to rebuild)")
            continue
        build_one(key, s, num_classes, device)
    print("\nDone. Models saved under", DEMO_DIR)