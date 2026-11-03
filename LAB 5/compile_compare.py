import time
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

x = torch.randn(1000, 100)

# Eager execution
start = time.time()

for _ in range(1000):
    y = model(x)

eager_time = time.time() - start

# Tracing
traced_model = torch.jit.trace(model, x)

start = time.time()

for _ in range(1000):
    y = traced_model(x)

traced_time = time.time() - start

print("Eager Time :", eager_time)
print("Traced Time:", traced_time)

print("\nTensor Information")
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Contiguous:", x.is_contiguous())

x_noncontig = x.t()

print("\nNon Contiguous Tensor")
print("Shape:", x_noncontig.shape)
print("Contiguous:", x_noncontig.is_contiguous())

start = time.time()

for _ in range(1000):
    z = x + 1

contiguous_time = time.time() - start

start = time.time()

for _ in range(1000):
    z = x_noncontig + 1

noncontiguous_time = time.time() - start

print("\nContiguous Time :", contiguous_time)
print("Non-Contiguous Time :", noncontiguous_time)