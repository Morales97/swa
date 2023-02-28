import os
import torch


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer, ema_opts=None, ema_interval=None):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data[0] * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()
        
        if ema_opts is not None and i % ema_interval == 0:
            for alpha in ema_opts.keys():
                ema_opts[alpha][i].update()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def eval(loader, model, criterion, epoch, name=''):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        loss_sum += loss.data[0] * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'Epoch': epoch,
        name + 'loss': loss_sum / len(loader.dataset),
        name + 'accuracy': correct / len(loader.dataset) * 100.0,
    }


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


class OptimizerEMA(object):
    '''
    EMA optimizer which can optionally apply EMA to BN statistics, with eman=True (see EMAN paper by Cai et al)
    '''
    def __init__(self, model, ema_model, alpha=0.999, eman=True, ramp_up=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.eman = eman
        self.step = 0
        self.ramp_up = ramp_up


    def update(self):
        if self.ramp_up:
            _alpha = min(self.alpha, (self.step + 1)/(self.step + 10)) 
        else:
            _alpha = self.alpha
        self.step += 1
        one_minus_alpha = 1.0 - _alpha

        # update learnable parameters
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.mul_(_alpha)
            ema_param.add_(param * one_minus_alpha)

        if self.eman:
            # update buffers (aka, non-learnable parameters). These are usually only BN stats
            for buffer, ema_buffer in zip(self.model.buffers(), self.ema_model.buffers()):
                if ema_buffer.dtype == torch.float32:      
                    ema_buffer.mul_(_alpha)
                    ema_buffer.add_(buffer * one_minus_alpha)