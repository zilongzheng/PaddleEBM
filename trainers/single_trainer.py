#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import copy
from typing import OrderedDict
import numpy as np

import logging
import datetime

import paddle
from paddle.distributed import ParallelEnv
from paddle.fluid.data import data

from datasets.single_image_dataset import SingleImageDataset
from models.builder import build_model
from utils.visual import tensor2img, save_image
from utils.filesystem import makedirs, save, load
from utils.timer import TimeAverager
from utils.logger import get_logger
from utils.visual import make_grid
from .builder import TRAINER
from .trainer import IterLoader


@TRAINER.register()
class SingleImageTrainer:

    def __init__(self, cfg):
        # base config
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.max_eval_steps = cfg.model.get('max_eval_steps', None)

        self.local_rank = ParallelEnv().local_rank
        self.log_interval = cfg.log_config.interval
        self.visual_interval = cfg.log_config.visiual_interval
        self.weight_interval = cfg.snapshot_config.interval

        self.current_scale = 1
        self.start_scale = 1
        self.global_steps = 0
        self.logger = get_logger()
        self.total_iters = cfg.total_iters


        # build metrics
        self.metrics = None
        validate_cfg = cfg.get('validate', None)
        if validate_cfg and 'metrics' in validate_cfg:
            self.metrics = self.model.setup_metrics(validate_cfg['metrics'])

        self.enable_visualdl = cfg.get('enable_visualdl', False)
        if self.enable_visualdl:
            import visualdl
            self.vdl_logger = visualdl.LogWriter(logdir=cfg.output_dir)

        # evaluate only
        if not cfg.is_train:
            return

        # build train dataloader
        data_cfg = cfg.dataset.train
        self.train_dataset = SingleImageDataset(**data_cfg)
        self.logger.info('Done Loading data [{}]'.format(cfg.dataset.train.dataroot))
        self.num_scales = self.train_dataset.num_scales
        self.by_epoch = False

        # build model
        # self.model = build_model(cfg.model)
        if 'params' in cfg.model:
            params = cfg.model.pop('params')
        else:
            params = OrderedDict()
        params.update({
            'max_image_size': self.train_dataset.max_image_size,
            'min_image_size': self.train_dataset.min_image_size,
            'num_scales': self.train_dataset.num_scales,
            'scale_factor': self.train_dataset.scale_factor,
            'scales': self.train_dataset.scales
        })
        cfg.model.params = params
        self.model = build_model(cfg.model)

        # build lr scheduler
        self.optim_cfg = cfg.optimizer

        # build optimizers
        # self.optimizers = self.model.setup_optimizers(cfg.optimizer)

        self.validate_interval = -1
        if cfg.get('validate', None) is not None:
            self.validate_interval = cfg.validate.get('interval', -1)

        self.time_count = {}
        self.best_metric = {}

    def distributed_data_parallel(self):
        strategy = paddle.distributed.prepare_context()
        for net_name, net in self.model.nets.items():
            self.model.nets[net_name] = paddle.DataParallel(net, strategy)

    def learning_rate_scheduler_step(self):
        if self.model.lr_scheduler:
            if isinstance(self.model.lr_scheduler, dict):
                for lr_scheduler in self.model.lr_scheduler.values():
                    if isinstance(lr_scheduler, paddle.optimizer.lr.LRScheduler):
                        lr_scheduler.step()
            elif isinstance(self.model.lr_scheduler,
                            paddle.optimizer.lr.LRScheduler):
                self.model.lr_scheduler.step()

    def train(self):
        batch_cost_averager = TimeAverager()
        self.current_scale = self.start_scale

        while self.current_scale < (self.num_scales + 1):

            # train single scale
            start_time = step_start_time = time.time()
            real_img = self.train_dataset[self.current_scale-1]

            self.logger.info('Working on scale {}: {}'.format(self.current_scale, real_img['img'].shape[1:]))

            self.model.setup_scale(self.current_scale)
            self.model.setup_input(real_img)
            optims = self.model.setup_optimizers(self.current_scale, self.optim_cfg)

            self.model.current_iter = self.current_iter = 1

            while self.current_iter < self.total_iters + 1:
    
                start_time = step_start_time = time.time()
                self.model.train_iter(optims)
                batch_cost_averager.record(time.time() - step_start_time,
                                        num_samples=self.cfg.get(
                                            'batch_size', 1))

                step_start_time = time.time()

                if self.current_iter % self.log_interval == 0:
                    self.step_time = batch_cost_averager.get_average()
                    self.ips = batch_cost_averager.get_ips_average()
                    self.print_log()

                    batch_cost_averager.reset()


                if self.current_iter == 1 or self.current_iter % self.visual_interval == 0:
                    self.visual('scale_%d'%self.current_scale)

                self.learning_rate_scheduler_step()

                self.current_iter += 1
            
            self.save(self.current_scale, 'netEBM%d'%self.current_scale, name='weight')

            self.current_scale += 1

    def test(self):
        if not hasattr(self, 'test_dataloader'):
            self.test_dataloader = build_dataloader(self.cfg.dataset.test,
                                                    is_train=False,
                                                    distributed=False)
        iter_loader = IterLoader(self.test_dataloader)
        if self.max_eval_steps is None:
            self.max_eval_steps = len(self.test_dataloader)

        if self.metrics:
            for metric in self.metrics.values():
                metric.reset()

        for i in range(self.max_eval_steps):
            data = next(iter_loader)
            self.model.setup_input(data)
            self.model.test_iter(metrics=self.metrics)

            visual_results = {}
            current_paths = self.model.get_image_paths()
            current_visuals = self.model.get_current_visuals()

            if len(current_visuals) > 0 and list(
                    current_visuals.values())[0].shape == 4:
                num_samples = list(current_visuals.values())[0].shape[0]
            else:
                num_samples = 1

            for j in range(num_samples):
                if j < len(current_paths):
                    short_path = os.path.basename(current_paths[j])
                    basename = os.path.splitext(short_path)[0]
                else:
                    basename = '{:04d}_{:04d}'.format(i, j)
                for k, img_tensor in current_visuals.items():
                    name = '%s_%s' % (basename, k)
                    if len(img_tensor.shape) == 4:
                        visual_results.update({name: img_tensor[j]})
                    else:
                        visual_results.update({name: img_tensor})

            self.visual('visual_test',
                        visual_results=visual_results,
                        step=self.batch_id,
                        is_save_image=True)

            if i % self.log_interval == 0:
                self.logger.info('Test iter: [%d/%d]' %
                                 (i, self.max_eval_steps))

        if self.metrics:
            for metric_name, metric in self.metrics.items():
                self.logger.info("Metric {}: {:.4f}".format(
                    metric_name, metric.accumulate()))

    def print_log(self):
        losses = self.model.get_current_losses()

        message = ''
        if self.by_epoch:
            message += 'Epoch: %d/%d, iter: %d/%d ' % (
                self.current_epoch, self.epochs, self.inner_iter,
                self.iters_per_epoch)
        else:
            message += 'Iter: %d/%d ' % (self.current_iter, self.total_iters)

        for opt_name, optim in self.model.optimizers.items():
            opt_lr = optim.get_lr()
            message += f'{opt_name} lr: {opt_lr:.3e} '

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
            if self.enable_visualdl:
                self.vdl_logger.add_scalar(k, v, step=self.global_steps)

        if hasattr(self, 'step_time'):
            message += 'batch_cost: %.5f sec ' % self.step_time

        if hasattr(self, 'data_time'):
            message += 'reader_cost: %.5f sec ' % self.data_time

        if hasattr(self, 'ips'):
            message += 'ips: %.5f images/s ' % self.ips

        if hasattr(self, 'step_time'):
            eta = self.step_time * (self.total_iters - self.current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            message += f'eta: {eta_str}'

        # print the message
        self.logger.info(message)        

    def visual(self,
               results_dir,
               visual_results=None,
               step=None,
               is_save_image=False):
        """
        visual the images, use visualdl or directly write to the directory
        Parameters:
            results_dir (str)     --  directory name which contains saved images
            visual_results (dict) --  the results images dict
            step (int)            --  global steps, used in visualdl
            is_save_image (bool)  --  weather write to the directory or visualdl
        """
        self.model.compute_visuals()

        if visual_results is None:
            visual_results = self.model.get_current_visuals()

        min_max = self.cfg.get('min_max', None)
        if min_max is None:
            min_max = (-1., 1.)

        image_num = self.cfg.get('image_num', None)
        if (image_num is None) or (not self.enable_visualdl):
            image_num = 1
        for label in list(visual_results.keys()):
            # image = make_grid(image, self.cfg.log_config.get('samples_every_row', 1)).detach()
            image = visual_results.pop(label)
            image_numpy = tensor2img(image, min_max, image_num)
            if (not is_save_image) and self.enable_visualdl:
                self.vdl_logger.add_image(
                    results_dir + '/' + label,
                    image_numpy,
                    step=step if step else self.global_steps,
                    dataformats="HWC" if image_num == 1 else "NCHW")
            else:
                if self.cfg.is_train:
                    if self.by_epoch:
                        msg = 'epoch%.3d_' % self.current_epoch
                    else:
                        msg = 'iter%.3d_' % self.current_iter
                else:
                    msg = ''
                makedirs(os.path.join(self.output_dir, results_dir))
                img_path = os.path.join(self.output_dir, results_dir,
                                        msg + '%s.png' % (label))
                save_image(image_numpy, img_path)

    def save(self, scale, net_name, opt_name=None, name='checkpoint'):
        save_dir = os.path.join(self.output_dir, 'models')
        if self.local_rank != 0:
            return

        assert name in ['checkpoint', 'weight']

        state_dicts = {}
        save_filename = 'scale_%s_%s.pdparams' % (scale, name)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_filename)
        # for net_name, net in self.model.nets.items():
        state_dicts[net_name] = self.model.nets[net_name].state_dict()

        state_dicts['scale'] = scale

        if name == 'weight':
            save(state_dicts, save_path)
            return


        if opt_name is not None:
            state_dicts[opt_name] = self.model.optimizers[opt_name].state_dict()

        save(state_dicts, save_path)

    def resume(self, checkpoint_path):
        state_dicts = load(checkpoint_path)
        if state_dicts.get('scale', None) is not None:
            self.start_scale = state_dicts['scale'] + 1
            # self.global_steps = self.iters_per_epoch * state_dicts['epoch']
        
        scale = self.start_scale - 1
        while scale > 0:
            # self.current_iter = state_dicts['epoch'] + 1
            for net_name, net in self.model.nets.items():
                if net_name in state_dicts:
                    net.set_state_dict(state_dicts[net_name])
                    self.logger.info(
                        'Loaded pretrained weight for net {}'.format(net_name))

                    self.save(scale, net_name, name='weight')

            if scale > 1:
                checkpoint_path = checkpoint_path.replace('scale_%d'%scale, 'scale_%d'%(scale-1))
                state_dicts = load(checkpoint_path)
            scale = scale - 1

    def load(self, weight_path):
        state_dicts = load(weight_path)

        for net_name, net in self.model.nets.items():
            if net_name in state_dicts:
                net.set_state_dict(state_dicts[net_name])
                self.logger.info(
                    'Loaded pretrained weight for net {}'.format(net_name))
            else:
                self.logger.warning(
                    'Can not find state dict of net {}. Skip load pretrained weight for net {}'
                    .format(net_name, net_name))

    def close(self):
        """
        when finish the training need close file handler or other.
        """
        if self.enable_visualdl:
            self.vdl_logger.close()
