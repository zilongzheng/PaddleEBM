import paddle
import paddle.nn as nn
import numpy as np
from collections import OrderedDict
from .base_model import BaseModel

from .builder import MODELS
from .criterions.builder import build_criterion

from solver.builder import build_optimizer, build_lr_scheduler
from utils.visual import make_grid
from .networks import define_generator, define_ebm


def gradient(tensor):
    if tensor._grad_ivar() is None:
        return None

    new_ivar = tensor._grad_ivar()
    return new_ivar.value().get_tensor()


@MODELS.register()
class CoopNets(BaseModel):
    """ This class implements the vanilla CoopNets model.

    vanilla CoopNets paper: https://arxiv.org/pdf/1609.09408
    """

    def __init__(self, generator, ebm, mcmc, params=None):
        super(CoopNets, self).__init__(params=params)
        self.mcmc_cfg = mcmc

        self.iter = 0
        self.lr_scheduler = OrderedDict()

        self.input_nz = generator['input_nz']
        # define generator
        self.nets['netG'] = define_generator(generator)
        self.nets['netEBM'] = define_ebm(ebm)

        self.netG_criterion = nn.loss.MSELoss(reduction='sum')

    def setup_input(self, input):
        self.inputs['obs'] = paddle.to_tensor(input['img'])

    def setup_optimizers(self, cfg):
        iters_per_epoch = cfg.pop('iters_per_epoch')
        for optim in cfg:
            opt_cfg = cfg[optim].copy()
            lr = opt_cfg.pop('learning_rate')
            if 'lr_scheduler' in opt_cfg:
                lr_cfg = opt_cfg.pop('lr_scheduler')
                lr_cfg['learning_rate'] = lr
                lr_cfg['iters_per_epoch'] = iters_per_epoch
                self.lr_scheduler[optim] = build_lr_scheduler(lr_cfg)
            else:
                self.lr_scheduler[optim] = lr
            cfg[optim] = opt_cfg
        self.optimizers['optimG'] = build_optimizer(
            cfg.optimG, self.lr_scheduler['optimG'], self.nets['netG'].parameters())
        self.optimizers['optimEBM'] = build_optimizer(
            cfg.optimEBM, self.lr_scheduler['optimEBM'], self.nets['netEBM'].parameters())

        return self.optimizers

    def mcmc_sample(self, init_state):
        cur_state = init_state.detach()
        for i in range(self.mcmc_cfg.num_steps):
            cur_state.stop_gradient = False
            neg_energy = self.nets['netEBM'](cur_state)
            grad = paddle.grad([neg_energy], [cur_state], retain_graph=True)[0]
            noise = paddle.rand(shape=self.inputs['obs'].shape)
            new_state = cur_state - self.mcmc_cfg.step_size * self.mcmc_cfg.step_size * \
                (cur_state / self.mcmc_cfg.refsig / self.mcmc_cfg.refsig -
                 grad) + self.mcmc_cfg.step_size * noise
            cur_state = new_state.detach()
        return cur_state

    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""

        batch_size = self.inputs['obs'].shape[0]
        self.z = paddle.rand(shape=(batch_size, self.input_nz, 1, 1))
        self.fake_gen = self.nets['netG'](self.z)
        self.fake_syn = self.mcmc_sample(self.fake_gen)

        self.visual_items['real'] = self.inputs['obs']
        self.visual_items['fake_gen'] = self.fake_gen
        self.visual_items['fake_syn'] = self.fake_syn

    def backward_EBM(self):
        self.real_neg_energy = self.nets['netEBM'](self.inputs['obs'])
        self.fake_neg_energy = self.nets['netEBM'](self.fake_syn)

        self.loss_EBM = paddle.sum(self.fake_neg_energy.mean(0) - self.real_neg_energy.mean(0))
        self.loss_EBM.backward()
        self.losses['loss_EBM'] = self.loss_EBM

    def backward_G(self):
        self.loss_G = self.netG_criterion(self.fake_gen, self.fake_syn)

        self.loss_G.backward()
        self.losses['loss_G'] = self.loss_G

    def train_iter(self, optims=None):
        self.forward()

        # update EBM
        self.set_requires_grad(self.nets['netEBM'], True)
        self.set_requires_grad(self.nets['netG'], False)
        self.optimizers['optimEBM'].clear_grad()
        self.backward_EBM()
        self.optimizers['optimEBM'].step()

        # update G
        self.set_requires_grad(self.nets['netG'], True)
        self.set_requires_grad(self.nets['netEBM'], False)
        self.optimizers['optimG'].clear_grad()
        self.backward_G()
        self.optimizers['optimG'].step()
