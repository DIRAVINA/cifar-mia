# src/train.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN


def train_model(model, X, y, epochs, device, lr=config.LEARNING_RATE,
                batch_size=config.BATCH_SIZE, weight_decay=0.0,
                X_val=None, y_val=None, verbose=True):
    """
    Trains a model on (X, y) and returns it.

    weight_decay is the L2 regularization strength (lambda in the paper).
    Keep it 0.0 for a normal (overfitted) model; we use it in the
    mitigation step later.

    If X_val / y_val are given, validation accuracy is printed too.
    """
    model.to(device)
    model.train()

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            msg = f"  epoch {epoch:3d}/{epochs}  loss {total_loss / len(X):.4f}"
            msg += f"  train acc {accuracy(model, X, y, device):.3f}"
            if X_val is not None:
                msg += f"  val acc {accuracy(model, X_val, y_val, device):.3f}"
            print(msg)

    return model


@torch.no_grad()
def predict_proba(model, X, device, temperature=1.0, batch_size=512):
    """
    Returns the model's prediction vectors: for each image, one probability
    per class (they sum to 1). THIS is what the attacker gets to see.

    temperature > 1 flattens the probabilities (a defense we test later).
    """
    model.to(device)
    model.eval()
    outputs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits = model(xb) / temperature
        outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def accuracy(model, X, y, device):
    """Fraction of images where the most likely class equals the true label."""
    probs = predict_proba(model, X, device)
    return float((probs.argmax(axis=1) == y).mean())


if __name__ == "__main__":
    # Self-test: run with  python -m src.train
    # Trains a quick target model and shows the train/test accuracy gap.
    set_seed()
    device = get_device()
    X, y = load_cifar()
    s = split_target_and_shadow_pool(X, y)

    num_classes = len(np.unique(y))
    model = CifarCNN(num_classes)

    print("Training target model (20 epochs for a quick test)...")
    train_model(model, s["X_target_train"], s["y_target_train"],
                epochs=20, device=device,
                X_val=s["X_target_test"], y_val=s["y_target_test"])

    tr = accuracy(model, s["X_target_train"], s["y_target_train"], device)
    te = accuracy(model, s["X_target_test"], s["y_target_test"], device)
    print(f"\nFinal train accuracy: {tr:.3f}")
    print(f"Final test accuracy : {te:.3f}")
    print(f"Gap (train - test)  : {tr - te:.3f}")