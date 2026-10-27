import torch
import torch.nn as nn
import time
from statistics import median

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

model = CNN()
model.eval()

times = []

for _ in range(20):

    x = torch.randn(128, 1, 28, 28)

    start = time.perf_counter()

    with torch.no_grad():
        model(x)

    end = time.perf_counter()

    times.append((end - start) * 1000)

print("Median Latency Batch 128 (ms):", median(times))