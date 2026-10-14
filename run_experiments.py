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
import time
from contextlib import contextmanager, redirect_stdout

import numpy as np
import torch

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN
from src.train import train_model
from src.shadow import build_attack_data, train_shadow_models, load_or_train_shadow_data
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


@contextmanager
def _results_lock():
    """Simple lock file, so scripts running at the same time take turns."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    lock = RESULTS_FILE + ".lock"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        os.remove(lock)


def update_result(key, fn):
    """
    Replaces results[key] with fn(old value or None) in results.json. The file
    is re-read under a lock right before writing, so scripts running at the
    same time do not overwrite each other's results.
    """
    with _results_lock():
        results = load_results()
        results[key] = fn(results.get(key))
        tmp = RESULTS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, RESULTS_FILE)


def save_result(key, value):
    """Stores one experiment under `key` in results.json."""
    update_result(key, lambda old: value)


def train_target(s, num_classes, device, l2=0.0, seed=config.SEED):
    torch.manual_seed(seed + config.TARGET_SEED_OFFSET)
    model = CifarCNN(num_classes)
    train_model(model, s["X_target_train"], s["y_target_train"],
                epochs=config.TARGET_EPOCHS, device=device,
                l2=l2, verbose=False)
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
            "accuracy": res["accuracy"],
            "median_class_precision": res["median_class_precision"],
            "per_class": res["per_class"]}


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
    return load_or_train_shadow_data(s, num_classes, device)


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
        target = train_target(s, num_classes, device, l2=lam)
        td = make_target_data(target, s, device)
        tr, te = split_acc(td)
        shadow = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"],
                                     num_classes, num_shadow=config.NUM_SHADOW_SWEEP,
                                     device=device, l2=lam)
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
    todo = {"sweep": exp_sweep, "mitigation": exp_mitigation, "l2": exp_l2}
    for key, fn in todo.items():
        if args.exp in (key, "all"):
            save_result(key, fn(X, y, num_classes, device))   # saved after each experiment
            print(f"\nSaved '{key}' results to {RESULTS_FILE}")