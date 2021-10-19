import paddle
from paddle.fluid.layers import tensor
from paddle.fluid.unique_name import generate
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from .base_model import BaseModel
import functools

from .builder import MODELS
from .criterions.builder import build_criterion

from solver.builder import build_optimizer, build_lr_scheduler
from .networks import define_ebm
from utils.visual import imresize_T
from PIL import Image


@MODELS.register()
class PatchGenCN(BaseModel):
    """ This class implements the PatchGenCN model.

    Patchwise Generative ConvNet: Training Energy-Based 
    Models from a Single Natural Image for Internal Learning
    https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Patchwise_Generative_ConvNet_Training_Energy-Based_Models_From_a_Single_Natural_CVPR_2021_paper.html
    """

    def __init__(self, nets, pad_size=5, params=None):
        super(PatchGenCN, self).__init__(params=params)

        self.lr_scheduler = OrderedDict()
        self.scales = self.params['scales']
        self.scale_factor = self.params['scale_factor']
        self.lambda_rec = self.params.get('lambda_rec', 0.1)

        self.mcmc_cfgs = {}
        for scale in range(1, self.params['num_scales']+1):
            sz = max(self.scales[scale-1])
            if scale == 1:
                net_cfg = nets[0].copy()
            elif sz < 64:
                net_cfg = nets[1].copy()
            else:
                net_cfg = nets[2].copy()
            mcmc_cfg = net_cfg.pop('mcmc')
            if 'noise_init' not in mcmc_cfg:
                mcmc_cfg['noise_init'] = 1.0
            if 'noise_min' not in mcmc_cfg:
                mcmc_cfg['noise_min'] = 0.0
            if 'noise_step' not in mcmc_cfg:
                mcmc_cfg['noise_step'] = 1000
            if 'num_steps_rec' not in mcmc_cfg:
                mcmc_cfg['num_steps_rec'] = mcmc_cfg['num_steps']

            self.nets['netEBM%d'%scale] = define_ebm(net_cfg)
            self.mcmc_cfgs['mcmc%d'%scale] = mcmc_cfg
        self.netEBM_criterion = nn.loss.MSELoss(reduction='sum')

        self.current_iter = 1

        self.pad_func = functools.partial(F.pad, pad=[pad_size, pad_size, pad_size, pad_size])
        self.unpad_func = lambda x : x[:, :, pad_size:-pad_size, pad_size:-pad_size]

    def setup_scale(self, scale):
        self.current_scale = scale

    def setup_input(self, input):
        self.inputs['obs%d'%self.current_scale] = paddle.to_tensor(input['img']).unsqueeze(0)

    def setup_optimizers(self, scale, cfg):
        self.optimizers.clear()
        iters_per_epoch = 1
        for optim in cfg:
            opt_cfg = cfg[optim].copy()
            # print(opt_cfg)
            lr = opt_cfg.pop('learning_rate')
            if 'lr_scheduler' in opt_cfg:
                self.lr_scheduler.clear()
                lr_cfg = opt_cfg.pop('lr_scheduler')
                lr_cfg['learning_rate'] = lr
                lr_cfg['iters_per_epoch'] = iters_per_epoch
                self.lr_scheduler[optim] = build_lr_scheduler(lr_cfg)
            else:
                self.lr_scheduler[optim] = lr
            # cfg[optim] = opt_cfg
            self.optimizers['%s%d'% (optim, scale)] = build_optimizer(
                opt_cfg, self.lr_scheduler[optim], self.nets['netEBM%d'% scale].parameters())

        return self.optimizers

    def mcmc_sample(self, net, init_state, noise_scale, cfg, mode="rand"):
        num_steps = cfg["num_steps"] if mode == "rand" else cfg["num_steps_rec"]
        cur_state = init_state.detach()
        for i in range(num_steps):
            cur_state.stop_gradient = False
            neg_energy = -net(cur_state)
            grad = paddle.grad([neg_energy], [cur_state], retain_graph=True)[0]
            noise = paddle.randn(shape=init_state.shape) * math.sqrt(cfg['step_size'])
            new_state = cur_state - 0.5 * cfg['step_size'] * grad + noise_scale * noise
            new_state = paddle.clip(new_state, -1.0, 1.0)
            cur_state = new_state.detach()
        return cur_state

    def generate_noise(self, shape):
        z = paddle.uniform([1, 1] + shape, dtype=paddle.float32)
        z = paddle.tile(z, [1, 3, 1, 1])
        return z

    def get_noise_scale(self, cfg):
        # noise_init = 1.0
        noise_scale = max((cfg['noise_init'] - cfg['noise_min']) * (1 - self.current_iter / (cfg['noise_step'] + 1.)), 0.) + cfg['noise_min']
        return noise_scale

    def get_init_fix(self, shape):
        init_img = self.inputs['obs1']
        ls_init = imresize_T(init_img, new_shape=[round(d*self.scale_factor) for d in shape], resample="lanczos")
        us_init = imresize_T(ls_init, new_shape=shape, resample="bicubic")
        return us_init


    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""
        if self.current_scale == 1:
            self.init_syn = self.generate_noise(list(self.scales[0]))
            if self.current_iter == 1:
                self.init_fix = self.get_init_fix(list(self.scales[0]))
        else:
            if self.current_iter == 1:
                self.init_fix = self.multi_scale_sequential_sample(self.current_scale, mode="fix")
            self.init_syn = self.multi_scale_sequential_sample(self.current_scale, mode="rand")

        init_syn_pad = self.pad_func(self.init_syn)
        init_fix_pad = self.pad_func(self.init_fix)

        # print(init_syn.shape)
        mcmc_cfg = self.mcmc_cfgs['mcmc%d'%self.current_scale]
        noise_scale = self.get_noise_scale(mcmc_cfg)
        self.fake_syn = self.mcmc_sample(self.nets['netEBM%d'%self.current_scale], init_syn_pad, noise_scale, mcmc_cfg)
        self.fake_rec = self.mcmc_sample(self.nets['netEBM%d'%self.current_scale], init_fix_pad, mcmc_cfg['noise_min'], mcmc_cfg, mode="rec")

        if self.current_iter == 1:
            self.visual_items['real'] = self.inputs['obs%d'%self.current_scale]
            self.visual_items['init_fix'] = self.init_fix
            self.visual_items['init_rand'] = self.init_syn
        self.visual_items['fake'] = self.unpad_func(self.fake_syn)
        self.visual_items['rec'] = self.unpad_func(self.fake_rec)
        self.losses['noise_scale'] = noise_scale

    def backward_EBM(self):
        self.real_neg_energy = self.nets['netEBM%d'%self.current_scale](self.pad_func(self.inputs['obs%d'%self.current_scale]))
        self.fake_neg_energy = self.nets['netEBM%d'%self.current_scale](self.fake_syn)
        self.reco_neg_energy = self.nets['netEBM%d'%self.current_scale](self.fake_rec)

        self.loss_EBM_syn = paddle.sum(self.fake_neg_energy.mean(
            0) - self.real_neg_energy.mean(0))
        self.loss_EBM_rec = paddle.sum(self.reco_neg_energy.mean(
            0) - self.real_neg_energy.mean(0))

        self.loss_EBM = self.loss_EBM_syn + self.lambda_rec * self.loss_EBM_rec

        self.loss_EBM.backward()
        self.losses['loss_EBM'] = self.loss_EBM

    def train_iter(self, optims=None):
        self.forward()

        # update EBM
        self.set_requires_grad(self.nets['netEBM%d'%self.current_scale], True)
        self.optimizers['optimEBM%d'%self.current_scale].clear_grad()
        self.backward_EBM()
        self.optimizers['optimEBM%d'%self.current_scale].step()

        self.current_iter += 1

    def multi_scale_sequential_sample(self, to_scale, mode="rand"):
        G_z = None
        if to_scale > 1:
            for scale in range(1, to_scale):
                img_h, img_w = self.scales[scale-1]

                if scale == 1:
                    if mode == "rand":
                        G_z = self.generate_noise([img_h, img_w])
                    else:
                        G_z = self.get_init_fix([img_h, img_w])

                G_z = self.pad_func(G_z)
                mcmc_cfg = self.mcmc_cfgs['mcmc%d'%scale]
                if mode == "rand":
                    G_z_next = self.mcmc_sample(self.nets['netEBM%d'%scale], G_z, mcmc_cfg['noise_min'], mcmc_cfg)
                else:
                    G_z_next = self.mcmc_sample(self.nets['netEBM%d'%scale], G_z, mcmc_cfg['noise_min'], mcmc_cfg, mode="rec")
                G_z_next = self.unpad_func(G_z_next)
                G_z_next = imresize_T(G_z_next, new_shape=self.scales[scale], resample="bicubic")
                G_z = G_z_next

        return G_z