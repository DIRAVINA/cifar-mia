# run_analysis.py
# Extra analysis for the dashboard. Run with:  python run_analysis.py
# Saves everything under the key "analysis" in results/results.json
import io
from contextlib import redirect_stdout

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.attack import train_attack_models, predict_membership_proba
from src.evaluate import evaluate_attack
from run_experiments import (train_target, make_target_data, load_results,
                             save_results, get_shadow_data_5000)


def fit_attack(shadow_data, num_classes, device):
    """Trains the per-class attack models without printing 10 lines each time."""
    with redirect_stdout(io.StringIO()):
        return train_attack_models(shadow_data, num_classes, device)


def histogram_data(td, bins=20):
    """
    Histograms (members vs non-members) of two signals the attack exploits:
      - confidence the target gives to the TRUE class
      - normalized prediction entropy (0 = certain, 1 = totally unsure)
    """
    probs, member = td["probs"], td["is_member"]
    conf = probs[np.arange(len(probs)), td["labels"]]
    ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / np.log(probs.shape[1])
    edges = np.linspace(0, 1, bins + 1)

    out = {}
    for name, values in [("true_class_confidence", conf), ("entropy", ent)]:
        m_counts, _ = np.histogram(values[member == 1], bins=edges)
        n_counts, _ = np.histogram(values[member == 0], bins=edges)
        out[name] = {"edges": edges.tolist(),
                     "members": m_counts.tolist(),
                     "non_members": n_counts.tolist()}
    return out


def roc_summary(true, score, points=200):
    """ROC curve (downsampled), AUC, and true-positive rate at low false-positive rates."""
    fpr, tpr, _ = roc_curve(true, score)
    idx = np.unique(np.linspace(0, len(fpr) - 1, points).astype(int))
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(),
            "auc": float(roc_auc_score(true, score)),
            "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
            "tpr_at_0.1pct_fpr": float(np.interp(0.001, fpr, tpr))}


def shadow_count_sweep(shadow_data, td, num_classes, device):
    """
    Re-trains the attack using only the first k shadow models (we slice the
    cached data, so NO shadow model is retrained). Each shadow contributes
    2 * SHADOW_TRAIN_SIZE rows: its members first, then its non-members.
    """
    per_shadow = 2 * config.SHADOW_TRAIN_SIZE
    available = len(shadow_data["probs"]) // per_shadow
    rows = []
    for k in [1, 2, 5, 10]:
        if k > available:
            continue
        print(f"  attack trained with {k} shadow model(s)...")
        part = {key: v[: k * per_shadow] for key, v in shadow_data.items()}
        models = fit_attack(part, num_classes, device)
        res = evaluate_attack(models, td, num_classes, device)
        score = predict_membership_proba(models, td["probs"], td["labels"], device)
        rows.append({"num_shadow": k, "precision": res["precision"],
                     "recall": res["recall"], "accuracy": res["accuracy"],
                     "auc": float(roc_auc_score(td["is_member"], score))})
        print(f"    precision {res['precision']:.3f} | recall {res['recall']:.3f} "
              f"| AUC {rows[-1]['auc']:.3f}")
    return rows


def per_class_gap(td, res, num_classes):
    """Per class: target's train accuracy, test accuracy, gap, and attack precision."""
    correct = td["probs"].argmax(axis=1) == td["labels"]
    rows = []
    for c in range(num_classes):
        m = td["labels"] == c
        tr = correct[m & (td["is_member"] == 1)].mean()
        te = correct[m & (td["is_member"] == 0)].mean()
        rows.append({"class": c, "train_acc": float(tr), "test_acc": float(te),
                     "gap": float(tr - te),
                     "precision": res["per_class"][c]["precision"]})
    return rows


if __name__ == "__main__":
    set_seed()
    device = get_device()
    X, y = load_cifar()
    s = split_target_and_shadow_pool(X, y)
    num_classes = len(np.unique(y))

    print("Training target model...")
    target = train_target(s, num_classes, device)
    td = make_target_data(target, s, device)

    shadow = get_shadow_data_5000(s, num_classes, device)
    print("Training attack models on all shadow data...")
    attack_models = fit_attack(shadow, num_classes, device)
    res = evaluate_attack(attack_models, td, num_classes, device)
    score = predict_membership_proba(attack_models, td["probs"], td["labels"], device)

    analysis = {}

    # 1) histograms
    analysis["histograms"] = histogram_data(td)

    # 2) ROC: shadow-model attack vs a simple no-shadow baseline
    baseline_score = td["probs"][np.arange(len(td["probs"])), td["labels"]]
    analysis["roc"] = {
        "shadow_attack": roc_summary(td["is_member"], score),
        "confidence_baseline": roc_summary(td["is_member"], baseline_score),
    }

    # 3) shadow-count sweep
    print("Shadow-count sweep:")
    analysis["shadow_sweep"] = shadow_count_sweep(shadow, td, num_classes, device)

    # 4) gap vs precision
    gap_rows = per_class_gap(td, res, num_classes)
    analysis["gap_vs_precision"] = {
        "rows": gap_rows,
        "correlation": float(np.corrcoef([r["gap"] for r in gap_rows],
                                         [r["precision"] for r in gap_rows])[0, 1]),
    }

    results = load_results()
    results["analysis"] = analysis
    save_results(results)

    a = analysis
    print("\n=== SUMMARY ===")
    print(f"Shadow attack AUC      : {a['roc']['shadow_attack']['auc']:.3f}")
    print(f"Confidence baseline AUC: {a['roc']['confidence_baseline']['auc']:.3f}")
    print(f"TPR at 1% FPR          : {a['roc']['shadow_attack']['tpr_at_1pct_fpr']:.3f}")
    print(f"Gap vs precision corr. : {a['gap_vs_precision']['correlation']:.3f}")
    print("Saved to results/results.json under 'analysis'")