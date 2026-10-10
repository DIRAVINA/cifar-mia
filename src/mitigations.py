# src/mitigations.py
# Output-side defenses from the paper (Section VIII). Each function takes the
# prediction vectors (N, num_classes) and returns a "defended" version, which
# is what the model would show the outside world.
import numpy as np


def no_mitigation(probs):
    return probs


def top_k(probs, k):
    """Keep only the k largest probabilities, zero out the rest."""
    out = np.zeros_like(probs)
    idx = np.argsort(-probs, axis=1)[:, :k]
    rows = np.arange(len(probs))[:, None]
    out[rows, idx] = probs[rows, idx]
    return out


def top1_label_only(probs):
    """Model reveals ONLY its predicted class (no probabilities at all)."""
    out = np.zeros_like(probs)
    out[np.arange(len(probs)), probs.argmax(axis=1)] = 1.0
    return out


def rounding(probs, d):
    """Round every probability to d decimal digits."""
    return np.round(probs, d)


def temperature(probs, t):
    """
    Softmax with temperature t. Dividing logits by t is the same as
    raising the probabilities to the power 1/t and renormalizing.
    A bigger t flattens the output (higher entropy).
    """
    z = np.log(np.clip(probs, 1e-12, 1.0)) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


MITIGATIONS = {
    "No mitigation": no_mitigation,
    "Top k=3": lambda p: top_k(p, 3),
    "Top k=1": lambda p: top_k(p, 1),
    "Top k=1 label only": top1_label_only,
    "Rounding d=3": lambda p: rounding(p, 3),
    "Rounding d=1": lambda p: rounding(p, 1),
    "Temperature t=5": lambda p: temperature(p, 5),
    "Temperature t=20": lambda p: temperature(p, 20),
}