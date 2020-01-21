import torch, os

class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(glow, nn_theta, optimizer, epoch, save_path):
    if not os.path.exists(save_path + 'snapshots/'):
        os.mkdir(save_path + 'snapshots/')

    torch.save({
        'glow_l3': glow.l3.state_dict(),
        'glow_l2': glow.l2.state_dict(),
        'glow_l1': glow.l1.state_dict(),
        'nn_theta_l3': nn_theta.l3.state_dict(),
        'nn_theta_l2': nn_theta.l2.state_dict(),
        'nn_theta_l1': nn_theta.l1.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_path+'snapshots/{}.pth'.format(epoch))



def track_grads(nn_theta, glow, iter):
    for name, param in nn_theta.l1.named_parameters():
        if param.requires_grad:
            writer.add_scalar('data/nntheta1', torch.max(param.grad.data), iter)
            writer.add_scalar('data/nntheta1', torch.min(param.grad.data), iter)

    for name, param in nn_theta.l2.named_parameters():
        if param.requires_grad:
            writer.add_scalar('data/nntheta2', torch.max(param.grad.data), iter)
            writer.add_scalar('data/nntheta2', torch.min(param.grad.data), iter)

    for name, param in nn_theta.l3.named_parameters():
        if param.requires_grad:
            writer.add_scalar('data/nntheta3', torch.max(param.grad.data), iter)
            writer.add_scalar('data/nntheta3', torch.min(param.grad.data), iter)

    for name, param in glow.l1.named_parameters():
        if param.requires_grad:
            writer.add_scalar('data/glow1', torch.max(param.grad.data), iter)
            writer.add_scalar('data/glow1', torch.min(param.grad.data), iter)

    for name, param in glow.l2.named_parameters():
        if param.requires_grad:
            writer.add_scalar('data/glow2', torch.max(param.grad.data), iter)
            writer.add_scalar('data/glow2', torch.min(param.grad.data), iter)

    for name, param in glow.l3.named_parameters():
        if param.requires_grad:
            writer.add_scalar('data/glow3', torch.max(param.grad.data), iter)
            writer.add_scalar('data/glow3', torch.min(param.grad.data), iter)
