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

# Batch size 64
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Model
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

model = MLP()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Task 6
accumulation_steps = 4

for epoch in range(5):

    model.train()

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(train_loader):

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss = loss / accumulation_steps

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:

            optimizer.step()

            optimizer.zero_grad()

    print(f"Epoch {epoch+1}/5 completed")

# Evaluation
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")