import torch
import torch.nn.utils.prune as prune

from baseline_cnn import BaselineCNN

model = BaselineCNN()

for amount in [0.10, 0.25, 0.50]:

    for module in model.modules():

        if isinstance(module, torch.nn.Conv2d):

            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0
            )

    print(f"Structured pruning {amount}")