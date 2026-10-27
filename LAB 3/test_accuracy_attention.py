import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

# Attention Model
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64, 10)

    def forward(self, x):

        batch_size = x.shape[0]

        # Split 28x28 image into sixteen 7x7 patches
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.contiguous().view(batch_size, 16, 49)

        # Patch embedding
        x = self.patch_embed(x)

        # Transformer
        x = self.transformer(x)

        # Mean pooling
        x = x.mean(dim=1)

        # Classification
        return self.fc(x)

model = AttentionModel()

