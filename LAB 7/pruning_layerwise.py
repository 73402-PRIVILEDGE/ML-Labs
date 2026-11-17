import torch
import torch.nn.utils.prune as prune

from baseline_cnn import BaselineCNN

model = BaselineCNN()

levels = [0, 0.25, 0.5, 0.75, 0.9, 0.95]

for level in levels:

    for module in model.modules():

        if isinstance(module, torch.nn.Conv2d):

            prune.l1_unstructured(
                module,
                name="weight",
                amount=level
            )

    print(f"Layerwise pruning {level}")