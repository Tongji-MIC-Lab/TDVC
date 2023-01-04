import torch
import torch.nn as nn


class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class AutomaticWeightedLoss(nn.Module):
    def __init__(self):
        super(AutomaticWeightedLoss, self).__init__()
        self.params = torch.nn.Parameter(torch.ones(2, requires_grad=True))

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum, self.params[0], self.params[1]
