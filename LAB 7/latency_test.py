import torch
import time

from baseline_cnn import BaselineCNN

model = BaselineCNN()
model.eval()

sample = torch.randn(1, 1, 28, 28)

runs = 100

start = time.time()

for _ in range(runs):
    model(sample)

end = time.time()

latency = (end - start) / runs

print("Latency:", latency * 1000, "ms")