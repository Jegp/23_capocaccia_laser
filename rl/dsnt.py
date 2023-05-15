from typing import Tuple

import torch


class DSNT(torch.nn.Module):
    def __init__(self, resolution: Tuple[int, int]):
        super().__init__()
        self.resolution = resolution
        self.probs_x = (
            torch.linspace(-1, 1, resolution[1]).repeat(resolution[0], 1).flatten()
        )
        self.probs_y = (
            torch.linspace(-1, 1, resolution[0]).repeat(resolution[1], 1).T.flatten()
        )

    def forward(self, x: torch.Tensor, s=None):
        if not x.device == self.probs_x.device:
            self.probs_x = self.probs_x.to(x.device)
            self.probs_y = self.probs_y.to(x.device)
        co_1 = (x.flatten(-2) * self.probs_x).sum(-1)
        co_2 = (x.flatten(-2) * self.probs_y).sum(-1)

        return torch.stack((co_2, co_1), -1), None
        
class DSNTLI(torch.nn.Module):
    def __init__(self, resolution: Tuple[int, int]):
        super().__init__()
        self.resolution = resolution
        self.probs_x = (
            torch.linspace(-1, 1, resolution[1]).repeat(resolution[0], 1).flatten()
        )
        self.probs_y = (
            torch.linspace(-1, 1, resolution[0]).repeat(resolution[1], 1).T.flatten()
        )
        self.li_tm = torch.nn.Parameter(torch.tensor([0.9, 0.9]))

    def forward(self, x: torch.Tensor, state=None):
        if not x.device == self.probs_x.device:
            self.probs_x = self.probs_x.to(x.device)
            self.probs_y = self.probs_y.to(x.device)
        co_1 = (x.flatten(-2) * self.probs_x).sum(-1)
        co_2 = (x.flatten(-2) * self.probs_y).sum(-1)

        cos = torch.stack((co_2, co_1), -1)
        if state is None:
            state = torch.zeros(2, device=x.device)

        out = []
        for t in cos:
            state = state - (state * self.li_tm) + t
            out.append(state.clone())

        return torch.stack(out), state