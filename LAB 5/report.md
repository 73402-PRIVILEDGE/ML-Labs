# Lab 5 Report: Automatic Differentiation and PyTorch Internals

## Task 1: Value Node Implementation

A custom `Value` node was implemented to represent nodes in a computational graph.

Each node stores:

- Data value
- Gradient value
- Parent nodes
- Local derivative function

The Value class allows mathematical operations to create a graph structure that supports automatic differentiation.

---

## Task 2: Mathematical Operations

The following operations were implemented:

- Addition (+)
- Multiplication (*)
- Power (**)
- ReLU activation

Each operation creates a new node in the computational graph and stores the local derivative required during backpropagation.

---

## Task 3: Backward Propagation

Backward propagation was implemented using:

- Topological sorting of graph nodes
- Reverse graph traversal
- Gradient accumulation through the chain rule

During the backward pass, gradients are propagated from the output node back through all dependent nodes, allowing each parameter to receive the correct gradient.

---

## Task 4: Verification with PyTorch Autograd

The custom automatic differentiation engine was compared against PyTorch's built-in autograd system.

### Custom Engine Results

```text
a = 42
b = 28
c = 14
```

### PyTorch Results

```text
a = 42
b = 28
c = 14
```

### Observation

The computed gradients matched exactly, confirming that the custom implementation produced correct forward and backward computations.

---

## Task 5: Neural Network Training

A binary classification neural network was implemented using PyTorch.

### Dataset

The moon-shaped classification dataset was generated using:

```python
make_moons(n_samples=1000, noise=0.2)
```

The features were standardized using `StandardScaler`, and the dataset was split into training and testing sets using an 80/20 ratio.

### Model Architecture

```text
Input Layer : 2 neurons
Hidden Layer: 8 neurons
Output Layer: 1 neuron
```

Network