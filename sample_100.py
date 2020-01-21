import torch
import os
import random
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.utils as utils
from torch.distributions.normal import Normal

from collections import namedtuple

from glow import _Glow, PreProcess, squeeze
from modules import *
from shell_util import AverageMeter, save_model
from optim_util import bits_per_dim
from dataloader import MovingObjects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_process = PreProcess().to(device)

Z_splits = namedtuple('Z_splits', 'l3 l2 l1')
Glow = namedtuple('Glow', 'l3 l2 l1')
NN_Theta = namedtuple('NNTheta', 'l3 l2 l1')


def flow_forward(x, flow):
    if x.min() < 0 or x.max() > 1:
        raise ValueError('Expected x in [0, 1], got min/max {}/{}'
                         .format(x.min(), x.max()))

    # pre-process
    x, sldj = pre_process(x)
    # L3
    x3 = squeeze(x, reverse=False)
    x3, sldj = flow.l3(x3, sldj, reverse=False)
    x3, x_split3 = x3.chunk(2, dim=1)
    # L2
    x2 = squeeze(x3, reverse=False)
    x2, sldj = flow.l2(x2, sldj, reverse=False)
    x2, x_split2 = x2.chunk(2, dim=1)
    # L1
    x1 = squeeze(x2, reverse=False)
    x1, sldj = flow.l1(x1, sldj)

    partition_out = Z_splits(l3=x_split3, l2=x_split2, l1=x1)
    partition_h = Z_splits(l3=x3, l2=x2, l1=None)

    return partition_out, partition_h, sldj


def sample_100(context, glow, nn_theta, temperature=0.5):
    for net in glow:
        net.eval()
    for net in nn_theta:
        net.eval()

    b_s = context.size(0)
    # generate two frames
    torchvision.utils.save_image(context[:, 0, ...].squeeze(), 'samples_100/context.png')

    for m in range(100):
        print(m)
        context_frame = context[:, 0, ...]
        for n in range(2):
            t0_zi, _, _ = flow_forward(context_frame, glow)

            mu_l1, logsigma_l1 = nn_theta.l1(t0_zi.l1)
            g1 = Normal(loc=mu_l1, scale=temperature * torch.exp(logsigma_l1))
            z1_sample = g1.sample()
            sldj = torch.zeros(b_s, device=device)

            # Inverse L1
            h1, sldj = glow.l1(z1_sample, sldj, reverse=True)
            h1 = squeeze(h1, reverse=True)

            # Sample z2
            mu_l2, logsigma_l2 = nn_theta.l2(t0_zi.l2, h1)
            g2 = Normal(loc=mu_l2, scale=temperature * torch.exp(logsigma_l2))
            z2_sample = g2.sample()
            h12 = torch.cat([h1, z2_sample], dim=1)
            h12, sldj = glow.l2(h12, sldj, reverse=True)
            h12 = squeeze(h12, reverse=True)

            # Sample z3
            mu_l3, logsigma_l3 = nn_theta.l3(t0_zi.l3, h12)
            g3 = Normal(loc=mu_l3, scale=temperature * torch.exp(logsigma_l3))
            z3_sample = g3.sample()

            x_t = torch.cat([h12, z3_sample], dim=1)
            x_t, sldj = glow.l3(x_t, sldj, reverse=True)
            x_t = squeeze(x_t, reverse=True)

            x_t = torch.sigmoid(x_t)

            if not os.path.exists('samples_100/'):
                os.mkdir('samples_100/')

            torchvision.utils.save_image(x_t, 'samples_100/sample{}_{}.png'.format(m, n+1))

            assert context_frame.shape == x_t.shape
            context_frame = x_t.clone()


def main():
    import torch.nn as nn

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tr = transforms.Compose([transforms.ToTensor()])
    train_data = MovingObjects("train", tr)
    train_loader = DataLoader(train_data,
                              num_workers=1,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True)

    in_chs = 3
    flow_l3 = nn.DataParallel(_Glow(in_channels=4 * in_chs, mid_channels=512, num_steps=24)).to(device)
    flow_l2 = nn.DataParallel(_Glow(in_channels=8 * in_chs, mid_channels=512, num_steps=24)).to(device)
    flow_l1 = nn.DataParallel(_Glow(in_channels=16 * in_chs, mid_channels=512, num_steps=24)).to(device)

    nntheta3 = nn.DataParallel(
        NNTheta(encoder_ch_in=4 * in_chs, encoder_mode='conv_net', h_ch_in=2 * in_chs,
                num_blocks=5)).to(device)  # z1:2x32x32
    nntheta2 = nn.DataParallel(
        NNTheta(encoder_ch_in=8 * in_chs, encoder_mode='conv_net', h_ch_in=4 * in_chs,
                num_blocks=5)).to(device)  # z2:4x16x16
    nntheta1 = nn.DataParallel(NNTheta(encoder_ch_in=16 * in_chs, encoder_mode='conv_net',
                                       num_blocks=5)).to(device)

    model_path = '/b_test/azimi/results/VideoFlow/SMovement/exp10/sacred/snapshots/109.pth'
    if True:
        print('model loading ...')
        flow_l3.load_state_dict(torch.load(model_path)['glow_l3'])
        flow_l2.load_state_dict(torch.load(model_path)['glow_l2'])
        flow_l1.load_state_dict(torch.load(model_path)['glow_l1'])
        nntheta3.load_state_dict(torch.load(model_path)['nn_theta_l3'])
        nntheta2.load_state_dict(torch.load(model_path)['nn_theta_l2'])
        nntheta1.load_state_dict(torch.load(model_path)['nn_theta_l1'])

    glow = Glow(l3=flow_l3, l2=flow_l2, l1=flow_l1)
    nn_theta = NN_Theta(l3=nntheta3, l2=nntheta2, l1=nntheta1)

    context = next(iter(train_loader)).cuda()
    sample_100(context, glow, nn_theta)


if __name__ == "__main__":
    main()
