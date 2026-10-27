import torch
import torch.nn as nn
from thop import profile

# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- CNN ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- Attention ----------------
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Linear(49, 64)

        encoder = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder,
            num_layers=2
        )

        self.fc = nn.Linear(64, 10)

    def forward(self, x):

        b = x.shape[0]

        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.contiguous().view(b, 16, 49)

        x = self.patch_embed(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.fc(x)


# ---------------- FLOPs & Parameters ----------------

dummy = torch.randn(1, 1, 28, 28)

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

for name, model in models.items():

    flops, params = profile(
        model,
        inputs=(dummy,),
        verbose=False
    )

    print("-" * 40)
    print(name)
    print(f"Parameters: {params:,}")
    print(f"FLOPs: {flops:,}")