import torch
import random
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as sched
import torch.nn.utils as utils
from torch.distributions.normal import Normal

from collections import namedtuple
from tqdm import tqdm

from glow import _Glow, PreProcess, squeeze
from modules import *
from shell_util import AverageMeter, save_model
from optim_util import bits_per_dim
from dataloader import MovingObjects

from tensorboardX import SummaryWriter
from sacred import Experiment
from sacred.observers import FileStorageObserver

global_step = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_process = PreProcess().to(device)

Z_splits = namedtuple('Z_splits', 'l3 l2 l1')
Glow = namedtuple('Glow', 'l3 l2 l1')
NN_Theta = namedtuple('NNTheta', 'l3 l2 l1')

PATH = './sacred/'
writer = SummaryWriter(PATH)
ex = Experiment()
ex.observers.append(FileStorageObserver.create(PATH))


@ex.config
def config():
    tr_conf = {
        'encoder_mode': 'conv_net',
        'enc_depth': 5,
        'n_epoch': 600,
        'b_s': 26,
        'lr': 1e-4,
        'k': 256,
        'input_channels': 3,
        'resume': True,
        'starting_epoch': 56
    }


def train_smovement(train_loader, glow, nn_theta, loss_fn, optimizer, scheduler, epoch):
    print("ID: exp12_1 testing lr 1e-4 and only one step movement, no glow loss with random patch")
    global global_step
    loss_meter = AverageMeter()
    # loss_fn_glow = GlowLoss()
    for net in glow:
        net.train()
    for net in nn_theta:
        net.train()

    with tqdm(total=len(train_loader.dataset)) as progress_bar:
        for itr, sequence in enumerate(train_loader):
            sequence = sequence.to(device)
            b_s = sequence.size(0)

            # start_index = torch.LongTensor(1).random_(0, 2)
            # random_patch = sequence[:, start_index:start_index + 2, :, :, :]

            random_patch = []
            for n in range(b_s):
               start_index = torch.LongTensor(1).random_(0, 2)
               random_patch.append(sequence[n, start_index:start_index + 2, :, :, :])
            random_patch = torch.stack(random_patch, dim=0)

            t0_zi, _, sldj_0 = flow_forward(random_patch[:, 0, :, :, :], glow)
            # z_glow = recover_z_shape(t0_zi)
            # loss_glow = loss_fn_glow(z_glow, sldj_0)

            t1_zi_out, t1_zi_h, sldj_1 = flow_forward(random_patch[:, 1, :, :, :], glow)
            h12 = t1_zi_h.l3

            mu_l3, logsigma_l3 = nn_theta.l3(t0_zi.l3, h12)
            g3 = Normal(loc=mu_l3, scale=torch.exp(logsigma_l3))

            h1 = t1_zi_h.l2
            mu_l2, logsigma_l2 = nn_theta.l2(t0_zi.l2, h1)
            g2 = Normal(loc=mu_l2, scale=torch.exp(logsigma_l2))

            mu_l1, logsigma_l1 = nn_theta.l1(t0_zi.l1)
            g1 = Normal(loc=mu_l1, scale=torch.exp(logsigma_l1))

            total_loss = loss_fn(g1, g2, g3, z=t1_zi_out, sldj=sldj_1,
                           input_dim=random_patch[:, 1, :, :, :].size())

            # total_loss = loss #+ loss_glow
            total_loss.backward()

            clip_grad_value(optimizer)

            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(global_step)

            loss_meter.update(total_loss.item(), b_s)
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=bits_per_dim(random_patch[:, 1, :, :, :], loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(b_s)
            global_step += 1

    print("global step:", global_step)

    torch.cuda.empty_cache()
    #save_model(glow, nn_theta, optimizer, scheduler, epoch, PATH)
    save_model(glow, nn_theta, optimizer, epoch, PATH)
    writer.add_scalar('data/train_loss', loss_meter.avg, epoch)
    writer.add_scalar('data/lr', get_lr(optimizer), epoch)

    context = next(iter(train_loader)).cuda()
    flow_inverse_smovement(context, glow, nn_theta, epoch)


def recover_z_shape(t_z):
    z1 = squeeze(t_z.l1, reverse=True)
    z2 = torch.cat([z1, t_z.l2], dim=1)
    z2 = squeeze(z2, reverse=True)
    z3 = torch.cat([z2, t_z.l3], dim=1)
    z3 = squeeze(z3, reverse=True)
    return z3


def clip_grad_value(optimizer, max_val=10.):
    for group in optimizer.param_groups:
        utils.clip_grad_value_(group['params'], max_val)


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


def flow_inverse_smovement(context, glow, nn_theta, epoch):
    for net in glow:
        net.eval()
    for net in nn_theta:
        net.eval()

    # pre-process the context frame
    b_s = context.size(0)
    context_frame = context[:, 0, ...]
    t0_zi, _, _ = flow_forward(context_frame, glow)

    mu_l1, logsigma_l1 = nn_theta.l1(t0_zi.l1)
    g1 = Normal(loc=mu_l1, scale=torch.exp(logsigma_l1))
    z1_sample = g1.sample()
    print("z1", z1_sample.shape)
    sldj = torch.zeros(b_s, device=device)

    # Inverse L1
    h1, sldj = glow.l1(z1_sample, sldj, reverse=True)
    h1 = squeeze(h1, reverse=True)

    # Sample z2
    mu_l2, logsigma_l2 = nn_theta.l2(t0_zi.l2, h1)
    g2 = Normal(loc=mu_l2, scale=torch.exp(logsigma_l2))
    z2_sample = g2.sample()
    h12 = torch.cat([h1, z2_sample], dim=1)
    h12, sldj = glow.l2(h12, sldj, reverse=True)
    h12 = squeeze(h12, reverse=True)

    # Sample z3
    mu_l3, logsigma_l3 = nn_theta.l3(t0_zi.l3, h12)
    g3 = Normal(loc=mu_l3, scale=torch.exp(logsigma_l3))
    z3_sample = g3.sample()

    x_t = torch.cat([h12, z3_sample], dim=1)
    x_t, sldj = glow.l3(x_t, sldj, reverse=True)
    x_t = squeeze(x_t, reverse=True)

    x_t = torch.sigmoid(x_t)

    torchvision.utils.save_image(x_t, 'samples/sample{}.png'.format(epoch))
    torchvision.utils.save_image(context[:, 0, ...].squeeze(), 'samples/context{}.png'.format(epoch))
    torchvision.utils.save_image(context[:, 1, ...].squeeze(), 'samples/gt{}.png'.format(epoch))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


@ex.automain
def main(tr_conf):
    import torch.nn as nn

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tr = transforms.Compose([transforms.ToTensor()])
    train_data = MovingObjects("train", tr, seed)
    train_loader = DataLoader(train_data,
                              num_workers=tr_conf['b_s'],
                              batch_size=tr_conf['b_s'],
                              shuffle=True,
                              pin_memory=True)

    param_list = []
    in_chs = tr_conf['input_channels']
    flow_l3 = nn.DataParallel(_Glow(in_channels=4 * in_chs, mid_channels=512, num_steps=24)).to(device)
    flow_l2 = nn.DataParallel(_Glow(in_channels=8 * in_chs, mid_channels=512, num_steps=24)).to(device)
    flow_l1 = nn.DataParallel(_Glow(in_channels=16 * in_chs, mid_channels=512, num_steps=24)).to(device)

    nntheta3 = nn.DataParallel(
        NNTheta(encoder_ch_in=4 * in_chs, encoder_mode=tr_conf['encoder_mode'], h_ch_in=2 * in_chs,
                num_blocks=tr_conf['enc_depth'])).to(device)  # z1:2x32x32
    nntheta2 = nn.DataParallel(
        NNTheta(encoder_ch_in=8 * in_chs, encoder_mode=tr_conf['encoder_mode'], h_ch_in=4 * in_chs,
                num_blocks=tr_conf['enc_depth'])).to(device)  # z2:4x16x16
    nntheta1 = nn.DataParallel(NNTheta(encoder_ch_in=16 * in_chs, encoder_mode=tr_conf['encoder_mode'],
                                       num_blocks=tr_conf['enc_depth'])).to(device)

    model_path = '/b_test/azimi/results/VideoFlow/SMovement/exp12_2/sacred/snapshots/55.pth'
    if tr_conf['resume']:
        print('model loading ...')
        flow_l3.load_state_dict(torch.load(model_path)['glow_l3'])
        flow_l2.load_state_dict(torch.load(model_path)['glow_l2'])
        flow_l1.load_state_dict(torch.load(model_path)['glow_l1'])
        nntheta3.load_state_dict(torch.load(model_path)['nn_theta_l3'])
        nntheta2.load_state_dict(torch.load(model_path)['nn_theta_l2'])
        nntheta1.load_state_dict(torch.load(model_path)['nn_theta_l1'])
        print("****LOAD THE OPTIMIZER")

    glow = Glow(l3=flow_l3, l2=flow_l2, l1=flow_l1)
    nn_theta = NN_Theta(l3=nntheta3, l2=nntheta2, l1=nntheta1)

    for f_level in glow:
        param_list += list(f_level.parameters())

    for nn in nn_theta:
        param_list += list(nn.parameters())

    loss_fn = NLLLossVF()

    optimizer = torch.optim.Adam(param_list, lr=tr_conf['lr'])
    optimizer.load_state_dict(torch.load(model_path)['optimizer'])
    optimizer.zero_grad()

    # scheduler_step = sched.StepLR(optimizer, step_size=1, gamma=0.99)
    # linear_decay = sched.LambdaLR(optimizer, lambda s: 1. - s / 150000. )
    # linear_decay.step(global_step)

    # scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / 10000))
    # optimizer.load_state_dict(torch.load(model_path)['optimizer'])

    for epoch in range(tr_conf['starting_epoch'], tr_conf['n_epoch']):
        print("the learning rate for epoch {} is {}".format(epoch, get_lr(optimizer)))
        train_smovement(train_loader, glow, nn_theta, loss_fn, optimizer, None, epoch)
        #scheduler_step.step()
