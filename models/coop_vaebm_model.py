import paddle
import paddle.nn as nn
import numpy as np
from collections import OrderedDict
from .base_model import BaseModel

from .builder import MODELS
from .criterions.builder import build_criterion

from solver.builder import build_optimizer, build_lr_scheduler
from .networks import define_generator, define_ebm, define_encoder


@MODELS.register()
class CoopVAEBM(BaseModel):
    """ This class implements the CoopVAEBM model.

    https://arxiv.org/pdf/2012.14936
    """

    def __init__(self, generator, ebm, encoder, mcmc, params=None):
        super(CoopVAEBM, self).__init__(params=params)
        self.mcmc_cfg = mcmc
        self.gen_cfg = generator

        self.lr_scheduler = OrderedDict()

        self.input_nz = generator['input_nz']
        # define generator
        self.nets['netG'] = define_generator(generator)
        self.nets['netEBM'] = define_ebm(ebm)

        encoder.input_sz = ebm.input_sz
        encoder.input_nc = ebm.input_nc
        encoder.output_nc = generator.input_nz
        self.nets['netEnc'] = define_encoder(encoder)

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
        self.optimizers['optimVAE'] = build_optimizer(
            cfg.optimVAE, self.lr_scheduler['optimVAE'], self.nets['netG'].parameters() + self.nets['netEnc'].parameters())
        self.optimizers['optimEBM'] = build_optimizer(
            cfg.optimEBM, self.lr_scheduler['optimEBM'], self.nets['netEBM'].parameters())

        return self.optimizers

    def get_z_random(self, batch_size, input_nz):
        random_type = self.gen_cfg.get('random_type', 'normal')
        if random_type == 'normal':
            return paddle.randn(shape=[batch_size, input_nz, 1, 1])
        elif random_type == ' uniform':
            return paddle.rand(shape=[batch_size, input_nz, 1, 1]) * 2.0 - 1.0
        else:
            raise NotImplementedError(
                'Unknown random type: {}'.format(random_type))

    def encode(self, input_image):
        mu, logVar = self.nets['netEnc'](input_image)
        std = paddle.exp(logVar * 0.5)
        eps = self.get_z_random(std.shape[0], std.shape[1])
        z = eps * std + mu
        return z, mu, logVar

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
        self.z = self.get_z_random(batch_size, self.input_nz)
        self.fake_gen = self.nets['netG'](self.z)
        self.fake_syn = self.mcmc_sample(self.fake_gen)

        self.syn_z, self.syn_mu, self.syn_logVar = self.encode(self.fake_syn)
        self.ae_res = self.nets['netG'](self.syn_z)

        self.visual_items['real'] = self.inputs['obs']
        self.visual_items['fake_gen'] = self.fake_gen
        self.visual_items['fake_syn'] = self.fake_syn

    def backward_EBM(self):
        self.real_neg_energy = self.nets['netEBM'](self.inputs['obs'])
        self.fake_neg_energy = self.nets['netEBM'](self.fake_syn)

        self.loss_EBM = paddle.sum(self.fake_neg_energy.mean(
            0) - self.real_neg_energy.mean(0))
        self.loss_EBM.backward()
        self.losses['loss_EBM'] = self.loss_EBM

    def backward_VAE(self):
        self.loss_recon = self.netG_criterion(self.fake_gen, self.fake_syn)
        self.loss_kl = paddle.sum(1 + self.syn_logVar - self.syn_mu.pow(
            2) - self.syn_logVar.exp()) * (-0.5 * self.params.lambda_kl)

        self.loss_VAE = self.loss_recon + self.params.lambda_kl * self.loss_kl

        self.loss_VAE.backward()
        self.losses['loss_recon'] = self.loss_recon
        self.losses['loss_kl'] = self.loss_kl

    def train_iter(self, optims=None):
        self.forward()

        # update EBM
        self.set_requires_grad(self.nets['netEBM'], True)
        self.set_requires_grad(self.nets['netG'], False)
        self.set_requires_grad(self.nets['netEnc'], False)
        self.optimizers['optimEBM'].clear_grad()
        self.backward_EBM()
        self.optimizers['optimEBM'].step()

        # update G
        self.set_requires_grad(self.nets['netG'], True)
        self.set_requires_grad(self.nets['netEnc'], True)
        self.set_requires_grad(self.nets['netEBM'], False)
        self.optimizers['optimVAE'].clear_grad()
        self.backward_VAE()
        self.optimizers['optimVAE'].step()
