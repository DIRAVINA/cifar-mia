# run_seeds.py
# Repeats the baseline attack with several seeds to measure run-to-run noise.
# Each seed gives a new target/shadow data split, a new target model, new
# shadow models and new attack models. The output defenses (top-k, rounding,
# temperature) are re-scored on every run too, since they need no retraining.
# Run with:  python run_seeds.py              (all seeds in config.REPEAT_SEEDS)
#            python run_seeds.py --seeds 1 2  (only some; several can run in parallel)
# Each finished seed is merged into the key "repeats" in results/results.json
import argparse
import io
from contextlib import redirect_stdout

import numpy as np

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.shadow import train_shadow_models
from src.attack import train_attack_models, predict_membership_proba
from src.evaluate import evaluate_attack
from src.mitigations import MITIGATIONS
from run_experiments import (train_target, make_target_data, split_acc,
                             update_result)
from run_analysis import roc_summary

METRICS = ["test_acc", "gap", "precision", "recall", "accuracy",
           "median_class_precision", "auc", "baseline_auc", "tpr_at_1pct_fpr"]


def one_run(X, y, num_classes, seed, device):
    s = split_target_and_shadow_pool(X, y, seed=seed)
    target = train_target(s, num_classes, device, seed=seed)
    td = make_target_data(target, s, device)
    shadow = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"], num_classes,
                                 num_shadow=config.NUM_SHADOW_REPEAT, device=device,
                                 seed=seed)
    run = {"seed": seed, "defenses": {}}
    for name, fn in MITIGATIONS.items():
        sd, tdm = dict(shadow), dict(td)
        sd["probs"], tdm["probs"] = fn(shadow["probs"]), fn(td["probs"])
        with redirect_stdout(io.StringIO()):
            models = train_attack_models(sd, num_classes, device, seed=seed)
        res = evaluate_attack(models, tdm, num_classes, device)
        tr, te = split_acc(tdm)
        row = {"train_acc": tr, "test_acc": te, "gap": tr - te,
               "precision": res["precision"], "recall": res["recall"],
               "accuracy": res["accuracy"],
               "median_class_precision": res["median_class_precision"]}
        if name == "No mitigation":
            score = predict_membership_proba(models, tdm["probs"], tdm["labels"], device)
            roc = roc_summary(tdm["is_member"], score)
            conf = tdm["probs"][np.arange(len(tdm["probs"])), tdm["labels"]]
            row.update({"auc": roc["auc"], "tpr_at_1pct_fpr": roc["tpr_at_1pct_fpr"],
                        "baseline_auc": roc_summary(tdm["is_member"], conf)["auc"]})
        run["defenses"][name] = row
    b = run["defenses"]["No mitigation"]
    print(f"  >> seed {seed}: test acc {b['test_acc']:.3f} | precision {b['precision']:.3f}"
          f" | recall {b['recall']:.3f} | AUC {b['auc']:.3f}")
    return run


def summarize_runs(runs):
    out = {}
    for name in MITIGATIONS:
        rows = [r["defenses"][name] for r in runs]
        out[name] = {m: {"mean": float(np.mean([r[m] for r in rows])),
                         "std": float(np.std([r[m] for r in rows], ddof=1))}
                     for m in METRICS if m in rows[0]}
    return out


def merge_run(old, run):
    """Adds one seed's run to the saved repeats and recomputes the summary."""
    order = {s: i for i, s in enumerate(config.REPEAT_SEEDS)}
    runs = [r for r in (old or {}).get("runs", []) if r["seed"] != run["seed"]] + [run]
    runs.sort(key=lambda r: order.get(r["seed"], len(order)))
    return {"seeds": [r["seed"] for r in runs], "num_shadow": config.NUM_SHADOW_REPEAT,
            "runs": runs, "summary": summarize_runs(runs) if len(runs) > 1 else {}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=config.REPEAT_SEEDS)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print("Device:", device)
    X, y = load_cifar()
    num_classes = len(np.unique(y))

    saved = {}
    for seed in args.seeds:
        print(f"\n##### seed {seed} #####")
        set_seed(seed)
        run = one_run(X, y, num_classes, seed, device)

        def merge(old, run=run):
            saved["repeats"] = merge_run(old, run)
            return saved["repeats"]
        update_result("repeats", merge)                 # saved after every seed

    rep = saved["repeats"]
    if not rep["summary"]:
        raise SystemExit("Only one seed saved so far; run the other seeds too.")
    print(f"\n=== mean +- std over seeds {rep['seeds']} ===")
    for name, row in rep["summary"].items():
        print(f"{name:20s} precision {row['precision']['mean']:.3f} +- {row['precision']['std']:.3f}"
              f" | recall {row['recall']['mean']:.3f} +- {row['recall']['std']:.3f}")
    b = rep["summary"]["No mitigation"]
    print(f"AUC {b['auc']['mean']:.3f} +- {b['auc']['std']:.3f} | baseline AUC "
          f"{b['baseline_auc']['mean']:.3f} +- {b['baseline_auc']['std']:.3f}")
    print("Saved to results/results.json under 'repeats'")
