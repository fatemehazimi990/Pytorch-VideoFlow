import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNTheta(nn.Module):
    def __init__(self, encoder_ch_in, encoder_mode, num_blocks, h_ch_in=None):
        super(NNTheta, self).__init__()
        self.encoder_mode = encoder_mode

        if h_ch_in is not None:
            self.conv1 = nn.Conv2d(in_channels=h_ch_in, out_channels=h_ch_in, kernel_size=1)
            initialize(self.conv1, mode='gaussian')

        dilations = [1, 2]
        self.latent_encoder = nn.ModuleList()
        for i in range(num_blocks):
            self.latent_encoder.append(nn.ModuleList(
                [self.latent_dist_encoder(encoder_ch_in, dilation=d, mode=encoder_mode) for d in dilations]))
        # print("latent encoder:", self.latent_encoder)

        if h_ch_in:
            self.conv2 = nn.Conv2d(in_channels=encoder_ch_in, out_channels=encoder_ch_in, kernel_size=1)
            initialize(self.conv2, mode='zeros')
        else:
            self.conv2 = nn.Conv2d(in_channels=encoder_ch_in, out_channels=2 * encoder_ch_in, kernel_size=1)
            initialize(self.conv2, mode='zeros')

    def forward(self, z_past, h=None):
        if h is not None:
            h = self.conv1(h)
            encoder_input = torch.cat([z_past, h], dim=1) 
        else:
            encoder_input = z_past.clone()

        for block in self.latent_encoder:
            parallel_outs = [pb(encoder_input) for pb in block]

            parallel_outs.append(encoder_input)
            encoder_input = sum(parallel_outs)

        last_t = self.conv2(encoder_input)
        deltaz_t, logsigma_t = last_t[:, 0::2, ...], last_t[:, 1::2, ...]

        # assert deltaz_t.shape == z_past.shape
        logsigma_t = torch.clamp(logsigma_t, min=-15., max=15.)
        mu_t = deltaz_t + z_past
        return mu_t, logsigma_t

    @staticmethod
    def latent_dist_encoder(ch_in, dilation, mode):

        if mode is "conv_net":
            layer1 = nn.Conv2d(in_channels=ch_in, out_channels=512, kernel_size=(3, 3),
                               dilation=(dilation, dilation), padding=(dilation, dilation))
            initialize(layer1, mode='gaussian')
            layer2 = GATU2D(channels=512)
            layer3 = nn.Conv2d(in_channels=512, out_channels=ch_in, kernel_size=(3, 3),
                               dilation=(dilation, dilation), padding=(dilation, dilation))
            initialize(layer3, mode='zeros')

            block = nn.Sequential(*[layer1, nn.ReLU(inplace=True), layer2, layer3])

        return block


class GATU2D(nn.Module):

    def __init__(self, channels):
        super(GATU2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        initialize(self.conv1, mode='gaussian')
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        initialize(self.conv2, mode='gaussian')

    def forward(self, x):
        out1 = torch.tanh(self.conv1(x))
        out2 = torch.sigmoid(self.conv2(x))
        return out1 * out2


class NLLLossVF(nn.Module):
    def __init__(self, k=256):
        super(NLLLossVF, self).__init__()
        self.k = k

    def forward(self, gaussian_1, gaussian_2, gaussian_3, z, sldj, input_dim):

        prior_ll_l3 = torch.sum(gaussian_3.log_prob(z.l3), [1, 2, 3])
        prior_ll_l2 = torch.sum(gaussian_2.log_prob(z.l2), [1, 2, 3])
        prior_ll_l1 = torch.sum(gaussian_1.log_prob(z.l1), [1, 2, 3])

        prior_ll = prior_ll_l1 + prior_ll_l2 + prior_ll_l3 - np.log(self.k) * np.prod(input_dim[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()
        return nll


class GlowLoss(nn.Module):
    def __init__(self, k=256):
        super(GlowLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
                   - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll


def initialize(layer, mode):
    if mode == 'gaussian':
        nn.init.normal_(layer.weight, 0., 0.05)
        nn.init.normal_(layer.bias, 0., 0.05)

    elif mode == 'zeros':
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

    else:
        raise NotImplementedError("To be implemented")
