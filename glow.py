import torch
import torch.nn as nn
import torch.nn.functional as F

from act_norm import ActNorm
from coupling import *
from inv_conv import InvConv


class _Glow(nn.Module):
    """Flow per level of Glow

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, in_channels, mid_channels, num_steps, num_levels=None):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for n,step in enumerate(self.steps):
                x, sldj = step(x, sldj, reverse)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels, coupling='affine', use_act_norm_in_coupling=True):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        if coupling is "additive":
            self.coup = AddCoupling(in_channels // 2, mid_channels, use_act_norm_in_coupling)
        else:
            self.coup = AffineCoupling(in_channels // 2, mid_channels)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, sldj, reverse)

        return x, sldj


class PreProcess(nn.Module):
    def __init__(self, bound=0.9):
        super(PreProcess, self).__init__()
        self.register_buffer('bounds', torch.tensor([bound], dtype=torch.float32))

    def forward(self, x):
        """Dequantize the input image `x` and convert to logits.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)
        #sldj = torch.zeros(x.shape[0]).cuda()

        return y, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x

if __name__ == "__main__":
    model = Glow(512, 3, 32)
    test_in = torch.rand(1,3,64,64)
    out = model(test_in)
    print(len(out[2]))
    for e in out[2]:
        print(e.shape)
