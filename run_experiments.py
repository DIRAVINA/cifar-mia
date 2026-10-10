# run_experiments.py
# Usage:
#   python run_experiments.py --exp mitigation
#   python run_experiments.py --exp l2
#   python run_experiments.py --exp sweep
#   python run_experiments.py --exp all
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
from src.shadow import (build_attack_data, train_shadow_models,
                        save_attack_data, load_attack_data)
from src.attack import train_attack_models
from src.evaluate import evaluate_attack
from src.mitigations import MITIGATIONS

RESULTS_FILE = os.path.join(config.RESULTS_DIR, "results.json")


# ---------- helpers ----------
def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def train_target(s, num_classes, device, weight_decay=0.0):
    torch.manual_seed(config.SEED + 999)
    model = CifarCNN(num_classes)
    train_model(model, s["X_target_train"], s["y_target_train"],
                epochs=config.TARGET_EPOCHS, device=device,
                weight_decay=weight_decay, verbose=False)
    return model


def make_target_data(model, s, device):
    probs, labels, member = build_attack_data(
        model, s["X_target_train"], s["y_target_train"],
        s["X_target_test"], s["y_target_test"], device)
    return {"probs": probs, "labels": labels, "is_member": member}


def split_acc(target_data):
    """Target's accuracy on its members (train) and non-members (test)."""
    correct = target_data["probs"].argmax(axis=1) == target_data["labels"]
    m = target_data["is_member"] == 1
    return float(correct[m].mean()), float(correct[~m].mean())


def run_attack(shadow_data, target_data, num_classes, device):
    buf = io.StringIO()                       # hide the per-class training prints
    with redirect_stdout(buf):
        attack_models = train_attack_models(shadow_data, num_classes, device)
    return evaluate_attack(attack_models, target_data, num_classes, device)


def summarize(res, train_acc, test_acc):
    return {"train_acc": train_acc, "test_acc": test_acc,
            "gap": train_acc - test_acc,
            "precision": res["precision"], "recall": res["recall"],
            "accuracy": res["accuracy"], "per_class": res["per_class"]}


def show(name, e):
    print(f"  >> {name}: train {e['train_acc']:.3f} | test {e['test_acc']:.3f} | "
          f"attack precision {e['precision']:.3f} | recall {e['recall']:.3f}")


# ---------- experiments ----------
def exp_sweep(X, y, num_classes, device):
    out = []
    for size in config.SWEEP_SIZES:
        print(f"\n##### SWEEP: target train size = {size} #####")
        s = split_target_and_shadow_pool(X, y, target_train_size=size)
        target = train_target(s, num_classes, device)
        td = make_target_data(target, s, device)
        tr, te = split_acc(td)
        shadow = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"],
                                     num_classes, num_shadow=config.NUM_SHADOW_SWEEP,
                                     train_size=size, device=device)
        e = summarize(run_attack(shadow, td, num_classes, device), tr, te)
        e["train_size"] = size
        show(f"size {size}", e)
        out.append(e)
    return out


def get_shadow_data_5000(s, num_classes, device):
    cache = os.path.join(config.RESULTS_DIR, "shadow_attack_data.npz")
    if os.path.exists(cache):
        print("Loading cached shadow data from", cache)
        return load_attack_data(cache)
    data = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"],
                               num_classes, device=device)
    save_attack_data(data, cache)
    return data


def exp_mitigation(X, y, num_classes, device):
    s = split_target_and_shadow_pool(X, y)
    target = train_target(s, num_classes, device)
    td = make_target_data(target, s, device)
    shadow = get_shadow_data_5000(s, num_classes, device)

    out = []
    for name, fn in MITIGATIONS.items():
        # The attacker is assumed to know the defense: shadows get it too.
        sd = dict(shadow)
        sd["probs"] = fn(shadow["probs"])
        tdm = dict(td)
        tdm["probs"] = fn(td["probs"])
        tr, te = split_acc(tdm)
        e = summarize(run_attack(sd, tdm, num_classes, device), tr, te)
        e["name"] = name
        show(name, e)
        out.append(e)
    return out


def exp_l2(X, y, num_classes, device):
    s = split_target_and_shadow_pool(X, y)
    out = []
    for lam in config.L2_LAMBDAS:
        print(f"\n##### L2 regularization: lambda = {lam} #####")
        target = train_target(s, num_classes, device, weight_decay=lam)
        td = make_target_data(target, s, device)
        tr, te = split_acc(td)
        shadow = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"],
                                     num_classes, num_shadow=config.NUM_SHADOW_SWEEP,
                                     device=device, weight_decay=lam)
        e = summarize(run_attack(shadow, td, num_classes, device), tr, te)
        e["lambda"] = lam
        show(f"lambda {lam}", e)
        out.append(e)
    return out


# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["sweep", "mitigation", "l2", "all"],
                        default="all")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print("Device:", device)
    X, y = load_cifar()
    num_classes = len(np.unique(y))
    results = load_results()

    todo = {"sweep": exp_sweep, "mitigation": exp_mitigation, "l2": exp_l2}
    for key, fn in todo.items():
        if args.exp in (key, "all"):
            results[key] = fn(X, y, num_classes, device)
            save_results(results)               # saved after each experiment
            print(f"\nSaved '{key}' results to {RESULTS_FILE}")