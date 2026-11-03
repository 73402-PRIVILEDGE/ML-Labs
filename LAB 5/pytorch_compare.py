import torch

# PyTorch tensors
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(1.0, requires_grad=True)

# Same expression used in our engine
y = ((a * b + c) ** 2).relu()

# Backward pass
y.backward()

print("Output:")
print(y.item())

print("\nGradients:")
print("a =", a.grad.item())
print("b =", b.grad.item())
print("c =", c.grad.item())