# src/attack.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config


class AttackMLP(nn.Module):
    """
    Small binary classifier: prediction vector -> "in" (1) or "out" (0).
    The paper uses a fully connected net with one hidden layer of size 64
    with ReLU, then a softmax output.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 2),      # 2 outputs: out / in
        )

    def forward(self, x):
        return self.net(x)


def _train_one(model, probs, is_member, epochs, device,
               lr=0.001, batch_size=config.ATTACK_BATCH_SIZE):
    """Trains a single attack model on (prediction vector -> member or not)."""
    model.to(device)
    model.train()
    ds = TensorDataset(torch.from_numpy(probs.astype(np.float32)),
                       torch.from_numpy(is_member.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def train_attack_models(data, num_classes, device, epochs=config.ATTACK_EPOCHS):
    """
    Trains ONE attack model PER CLASS (the paper's design).

    Why per class? The target's output pattern depends heavily on the true
    class, so a separate model for each class is more accurate.

    data: dict with 'probs', 'labels', 'is_member' (from shadow models)
    Returns: dict {class_id: trained AttackMLP}
    """
    attack_models = {}
    for c in range(num_classes):
        mask = data["labels"] == c               # rows whose TRUE label is class c
        model = AttackMLP(num_classes)
        _train_one(model, data["probs"][mask], data["is_member"][mask], epochs, device)
        attack_models[c] = model
        print(f"  attack model for class {c}: trained on {int(mask.sum())} rows")
    return attack_models


@torch.no_grad()
def predict_membership(attack_models, probs, labels, device):
    """
    For each (prediction vector, true label), picks the attack model for that
    label and predicts 1 = "member", 0 = "non-member".
    """
    preds = np.zeros(len(probs), dtype=np.int64)
    for c, model in attack_models.items():
        mask = labels == c
        if mask.sum() == 0:
            continue
        model.to(device)
        model.eval()
        x = torch.from_numpy(probs[mask].astype(np.float32)).to(device)
        preds[mask] = model(x).argmax(dim=1).cpu().numpy()
    return preds




@torch.no_grad()
def predict_membership_proba(attack_models, probs, labels, device):
    """
    Same idea as predict_membership(), but returns the attack's SCORE:
    the probability (0 to 1) that each record was in the training set.
    Needed for ROC curves and histograms.
    """
    scores = np.zeros(len(probs), dtype=np.float32)
    for c, model in attack_models.items():
        mask = labels == c
        if mask.sum() == 0:
            continue
        model.to(device)
        model.eval()
        x = torch.from_numpy(probs[mask].astype(np.float32)).to(device)
        scores[mask] = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
    return scores