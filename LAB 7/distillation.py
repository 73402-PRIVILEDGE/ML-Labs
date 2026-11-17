import torch
import torch.nn as nn
import torch.nn.functional as F

T = 4
alpha = 0.5

def distillation_loss(
        student_logits,
        teacher_logits,
        labels):

    ce_loss = F.cross_entropy(
        student_logits,
        labels
    )

    kl_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    )

    loss = alpha * ce_loss + \
        (1 - alpha) * (T * T) * kl_loss

    return loss

print("Knowledge Distillation Ready")