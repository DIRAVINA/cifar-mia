# src/model.py
import torch
import torch.nn as nn


class CifarCNN(nn.Module):
    """
    The CNN described in the paper for CIFAR:
      2 x (convolution -> Tanh -> max pooling), then a fully connected
      layer of size 128, then a softmax output layer.

    Note: this model outputs raw scores ("logits"). The softmax that turns
    them into probabilities is applied later in predict_proba(), because
    PyTorch's loss function expects logits during training.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),   # 32x32 -> 28x28
            nn.Tanh(),
            nn.MaxPool2d(2),                   # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=5),  # 14x14 -> 10x10
            nn.Tanh(),
            nn.MaxPool2d(2),                   # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 64 * 5 * 5 = 1600 numbers
            nn.Linear(64 * 5 * 5, 128),
            nn.Tanh(),
            nn.Linear(128, num_classes),       # logits, one per class
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    # Self-test: run with  python -m src.model
    model = CifarCNN(num_classes=10)
    fake_batch = torch.randn(4, 3, 32, 32)   # 4 fake images
    out = model(fake_batch)
    print("Output shape:", out.shape)         # expect (4, 10)
    print("Parameters:", sum(p.numel() for p in model.parameters()))