import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Same model as baseline
class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = BigCNN()

parameters_to_prune = (
    (model.features[0], "weight"),
    (model.features[2], "weight"),
    (model.features[5], "weight"),
    (model.classifier[1], "weight"),
    (model.classifier[3], "weight"),
)

sparsities = [0, 0.25, 0.5, 0.75, 0.9, 0.95]

for sparsity in sparsities:

    test_model = BigCNN()

    prune.global_unstructured(
        (
            (test_model.features[0], "weight"),
            (test_model.features[2], "weight"),
            (test_model.features[5], "weight"),
            (test_model.classifier[1], "weight"),
            (test_model.classifier[3], "weight"),
        ),
        pruning_method=prune.L1Unstructured,
        amount=sparsity
    )

    print(f"Sparsity: {sparsity*100:.0f}%")