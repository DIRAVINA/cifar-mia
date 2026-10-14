# run_dp_experiments.py
# DP-SGD as a defense. Run with:  python run_dp_experiments.py
# Saves results under the key "dp_sgd" in results/results.json
import numpy as np

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.dp import train_model_dp
from src.shadow import build_attack_data
from run_experiments import (make_target_data, split_acc, run_attack,
                             summarize, show, save_result)


def train_shadow_models_dp(X_pool, y_pool, num_classes, epsilon, device,
                           num_shadow, train_size=config.SHADOW_TRAIN_SIZE,
                           seed=config.SEED):
    """
    Same recipe as train_shadow_models() in shadow.py, but every shadow model
    is trained with DP-SGD at the same epsilon as the target. (The attacker is
    assumed to know the defense, which is the stricter test.)
    """
    rng = np.random.RandomState(seed)
    probs_all, labels_all, member_all = [], [], []
    for i in range(num_shadow):
        print(f"[DP shadow {i + 1}/{num_shadow}] training...")
        idx = rng.permutation(len(X_pool))
        tr_idx, te_idx = idx[:train_size], idx[train_size: 2 * train_size]
        X_tr, y_tr = X_pool[tr_idx], y_pool[tr_idx]
        X_te, y_te = X_pool[te_idx], y_pool[te_idx]

        model, _ = train_model_dp(X_tr, y_tr, num_classes, epsilon, device,
                                  seed=seed + i)
        probs, labels, member = build_attack_data(model, X_tr, y_tr, X_te, y_te, device)
        probs_all.append(probs)
        labels_all.append(labels)
        member_all.append(member)

    return {"probs": np.concatenate(probs_all), "labels": np.concatenate(labels_all),
            "is_member": np.concatenate(member_all)}


def exp_dp(X, y, num_classes, device):
    s = split_target_and_shadow_pool(X, y)
    out = []
    for eps in config.DP_EPSILONS:
        print(f"\n##### DP-SGD: target epsilon = {eps} #####")
        target, spent = train_model_dp(s["X_target_train"], s["y_target_train"],
                                       num_classes, eps, device,
                                       seed=config.SEED + config.TARGET_SEED_OFFSET)
        print(f"  target trained, epsilon spent = {spent:.2f}")
        td = make_target_data(target, s, device)
        tr, te = split_acc(td)

        shadow = train_shadow_models_dp(s["X_shadow_pool"], s["y_shadow_pool"],
                                        num_classes, eps, device,
                                        num_shadow=config.NUM_SHADOW_SWEEP)
        e = summarize(run_attack(shadow, td, num_classes, device), tr, te)
        e.update({"epsilon": eps, "epsilon_spent": spent, "name": f"DP-SGD eps={eps:g}"})
        show(e["name"], e)
        out.append(e)
    return out


if __name__ == "__main__":
    set_seed()
    device = get_device()
    print("Device:", device)
    X, y = load_cifar()
    num_classes = len(np.unique(y))

    save_result("dp_sgd", exp_dp(X, y, num_classes, device))
    print("\nSaved 'dp_sgd' results to results/results.json")