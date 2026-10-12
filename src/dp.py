# src/dp.py
# Differentially private training (DP-SGD) using the Opacus library.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN
from src.train import accuracy


def train_model_dp(X, y, num_classes, epsilon, device, epochs=config.DP_EPOCHS,
                   delta=config.DP_DELTA, max_grad_norm=config.DP_MAX_GRAD_NORM,
                   lr=config.DP_LEARNING_RATE, batch_size=config.DP_BATCH_SIZE,
                   seed=config.SEED):
    """
    Trains a CifarCNN with DP-SGD so that the whole training run satisfies
    (epsilon, delta)-differential privacy.

    How it works: for every batch, the gradient of EACH example is clipped to
    max_grad_norm, the clipped gradients are summed, and random noise is added.
    Opacus picks the noise level so the total privacy cost over all epochs
    equals the target epsilon.

    Returns: (clean CifarCNN model, epsilon actually spent)
    """
    torch.manual_seed(seed)
    base = CifarCNN(num_classes).to(device)
    optimizer = torch.optim.Adam(base.parameters(), lr=lr)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    engine = PrivacyEngine()
    model, optimizer, loader = engine.make_private_with_epsilon(
        module=base, optimizer=optimizer, data_loader=loader,
        target_epsilon=epsilon, target_delta=delta,
        epochs=epochs, max_grad_norm=max_grad_norm)

    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            if len(xb) == 0:          # DP sampling can (rarely) give an empty batch
                continue
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    spent = float(engine.get_epsilon(delta))

    # Copy the trained weights into a plain CifarCNN, so the rest of our code
    # (predict_proba, accuracy, ...) works unchanged.
    clean = CifarCNN(num_classes)
    clean.load_state_dict(base.state_dict())
    return clean, spent


if __name__ == "__main__":
    # Quick self-test: python -m src.dp   (3 epochs only)
    set_seed()
    device = get_device()
    X, y = load_cifar()
    s = split_target_and_shadow_pool(X, y)
    nc = len(np.unique(y))

    print("Quick DP-SGD test: epsilon = 10, 3 epochs")
    model, spent = train_model_dp(s["X_target_train"], s["y_target_train"], nc,
                                  epsilon=10.0, device=device, epochs=3)
    tr = accuracy(model, s["X_target_train"], s["y_target_train"], device)
    te = accuracy(model, s["X_target_test"], s["y_target_test"], device)
    print(f"epsilon spent : {spent:.2f}")
    print(f"train accuracy: {tr:.3f}")
    print(f"test accuracy : {te:.3f}")
    print(f"gap           : {tr - te:.3f}")