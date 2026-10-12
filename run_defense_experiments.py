# run_defense_experiments.py
# Dropout (and dropout + L2) as defenses. Run with:
#   python run_defense_experiments.py
# Saves results under the key "dropout" in results/results.json
import numpy as np
import torch

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN
from src.train import train_model
from src.shadow import train_shadow_models
from run_experiments import (make_target_data, split_acc, run_attack,
                             summarize, show, load_results, save_results)


def train_target_reg(s, num_classes, device, dropout=0.0, weight_decay=0.0):
    """Trains the target with dropout and/or L2 regularization."""
    torch.manual_seed(config.SEED + 999)
    model = CifarCNN(num_classes, dropout=dropout)
    train_model(model, s["X_target_train"], s["y_target_train"],
                epochs=config.TARGET_EPOCHS, device=device,
                weight_decay=weight_decay, verbose=False)
    return model


def exp_dropout(X, y, num_classes, device):
    s = split_target_and_shadow_pool(X, y)
    out = []
    for name, p, lam in config.DROPOUT_SETTINGS:
        print(f"\n##### {name} #####")
        target = train_target_reg(s, num_classes, device, dropout=p, weight_decay=lam)
        td = make_target_data(target, s, device)
        tr, te = split_acc(td)

        # The attacker knows the defense: shadows are trained the same way.
        shadow = train_shadow_models(s["X_shadow_pool"], s["y_shadow_pool"],
                                     num_classes, num_shadow=config.NUM_SHADOW_SWEEP,
                                     device=device, weight_decay=lam, dropout=p)
        e = summarize(run_attack(shadow, td, num_classes, device), tr, te)
        e.update({"name": name, "dropout": p, "lambda": lam})
        show(name, e)
        out.append(e)
    return out


if __name__ == "__main__":
    set_seed()
    device = get_device()
    print("Device:", device)
    X, y = load_cifar()
    num_classes = len(np.unique(y))

    results = load_results()
    results["dropout"] = exp_dropout(X, y, num_classes, device)
    save_results(results)
    print("\nSaved 'dropout' results to results/results.json")