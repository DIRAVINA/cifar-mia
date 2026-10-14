# src/train.py
import numpy as np
import torch
import torch.nn as nn

import config
from src.data import get_device, set_seed, load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN


def minibatches(X, y, batch_size, shuffle=True):
    """
    Yields (X, y) mini-batches of tensors that already sit on the device.
    Same as a shuffled DataLoader over a TensorDataset, but it slices whole
    batches at once instead of fetching and stacking samples one by one,
    which is far faster for small models.
    """
    n = len(X)
    order = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        b = order[i:i + batch_size]
        yield X[b], y[b]


def train_model(model, X, y, epochs, device, lr=config.LEARNING_RATE,
                batch_size=config.BATCH_SIZE, l2=0.0,
                X_val=None, y_val=None, verbose=True):
    """
    Trains a model on (X, y) and returns it.

    l2 is the L2 regularization strength, lambda in the paper (Section VIII):
    the loss becomes  cross-entropy + lambda * sum(theta^2)  over all
    parameters. Keep it 0.0 for a normal (overfitted) model.

    Note: the penalty is added to the loss on purpose. Adam's own
    `weight_decay=lambda` adds lambda * theta to the gradient, which equals a
    penalty of only lambda / 2 * sum(theta^2), i.e. half the paper's lambda.

    If X_val / y_val are given, validation accuracy is printed too.
    """
    model.to(device)
    model.train()

    # Data goes to the device once, not batch by batch.
    X_t, y_t = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    params = list(model.parameters())

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in minibatches(X_t, y_t, batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            if l2 > 0:
                loss = loss + l2 * sum(p.pow(2).sum() for p in params)
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