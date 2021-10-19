import paddle
import paddle.nn as nn
import functools

from paddle.nn import BatchNorm2D
from .modules.norm import build_norm_layer
from .modules.init import init_weights
from .modules.nn import Spectralnorm

def define_generator(netG_cfg):
    _cfg = netG_cfg.copy()
    net_name = _cfg.pop('name')
    netG = None
    if net_name == 'DCGenerator':
        netG = DCGenerator(**_cfg)
    elif net_name == 'MNISTGenerator':
        netG = MNISTGenerator(**_cfg)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_name)
    init_cfgs = {k:v for k, v in _cfg.items() if k in ['init_type', 'init_gain', 'distribution']}
    init_weights(netG, **init_cfgs)
    return netG

def define_ebm(netEBM_cfg):
    _cfg = netEBM_cfg.copy()
    net_name = _cfg.pop('name')
    netEBM = None
    if net_name == 'DCEBM':
        netEBM = DCEBM(**_cfg)
    elif net_name == 'PatchEBM':
        netEBM = PatchEBM(**_cfg)
    else:
        raise NotImplementedError('EBM name [%s] is not recognized' % net_name)
    init_cfgs = {k:v for k, v in _cfg.items() if k in ['init_type', 'init_gain', 'distribution']}
    init_weights(netEBM, **init_cfgs)
    return netEBM
    

class DCEBM(nn.Layer):
    """ Vanilla Deep Convolutional EBM
    """
    def __init__(self,
                 input_sz,
                 input_nc,
                 output_nc,
                 nef=64,
                 **kwargs):
        """Construct a DCGenerator generator
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(DCEBM, self).__init__()

        model = [
            nn.Conv2D(input_nc, nef, 4, 2, 1),
            nn.LeakyReLU(),

            nn.Conv2D(nef, nef * 2, 4, 2, 1),
            nn.LeakyReLU(),

            nn.Conv2D(nef * 2, nef * 4, 4, 2, 1),
            nn.LeakyReLU()
        ]

        self.conv = nn.Sequential(*model)
        out_sz = input_sz // 8
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_sz * out_sz * nef * 4, output_nc)
        )

    def forward(self, x):
        """Standard forward"""
        out = self.conv(x)
        return self.fc(out).sum()

class PatchEBM(nn.Layer):
    """ Patchwise Generative ConvNet
    https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Patchwise_Generative_ConvNet_Training_Energy-Based_Models_From_a_Single_Natural_CVPR_2021_paper.html
    """
    def __init__(self,
                 input_nc,
                 output_nc,
                 nefs=[64, 64, 64],
                 sn=True,
                 init_gain=0.002,
                 **kwargs):
        """Construct a DCGenerator generator
        Args:
            input_sz (int)      -- the number of dimension in input images
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(PatchEBM, self).__init__()

        weight_attr = paddle.framework.ParamAttr(initializer=nn.initializer.Normal(std=init_gain))

        model = []

        i_c = input_nc
        for l_i, nef in enumerate(nefs):
            o_c = output_nc if l_i == len(nefs)-1 else nef
            layer = nn.Conv2D(i_c, o_c, 3, 1, weight_attr=weight_attr)
            i_c = o_c
            if sn:
                layer = Spectralnorm(layer)
            model.append(layer)
            model.append(nn.ELU())

        self.conv = nn.Sequential(*model)
        self.fc = nn.Sequential(
            nn.Flatten(),
        )

        # self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                m.weight.data.normal_(0, 0.005)

    def forward(self, x):
        """Standard forward"""
        out = self.conv(x)
        return self.fc(out).sum()



class MNISTGenerator(nn.Layer):
    """ Deep Convolutional Generator for MNIST
    """
    def __init__(self,
                 input_nz,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_type='batch',
                 **kwargs):
        """Construct a DCGenerator generator
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(MNISTGenerator, self).__init__()

        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        model = [
            nn.Conv2DTranspose(input_nz, ngf * 8, 4, 1, 0),
            norm_layer(ngf * 8),
            nn.ReLU(),

            nn.Conv2DTranspose(ngf * 8, ngf * 4, 4, 1, 0),
            norm_layer(ngf * 4),
            nn.ReLU(),

            nn.Conv2DTranspose(ngf * 4, ngf * 2, 4, 2, 1),
            norm_layer(ngf * 2),
            nn.ReLU(),

            nn.Conv2DTranspose(ngf * 2, output_nc, 4, 2, 1),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)

class DCGenerator(nn.Layer):
    """ Deep Convolutional Generator
    """
    def __init__(self,
                 input_nz,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_type='batch',
                 **kwargs):
        """Construct a DCGenerator generator
        Args:
            input_nz (int)      -- the number of dimension in input noise
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(DCGenerator, self).__init__()

        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        mult = 8
        n_downsampling = 4

        if norm_type == 'batch':
            model = [
                nn.Conv2DTranspose(input_nz,
                                    ngf * mult,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0,
                                    bias_attr=use_bias),
                BatchNorm2D(ngf * mult),
                nn.ReLU()
            ]
        else:
            model = [
                nn.Conv2DTranspose(input_nz,
                                    ngf * mult,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0,
                                    bias_attr=use_bias),
                norm_layer(ngf * mult),
                nn.ReLU()
            ]

        for i in range(1,n_downsampling):  # add upsampling layers
            mult = 2**(n_downsampling - i)
            output_size = 2**(i+2)
            if norm_type == 'batch':
                model += [
                nn.Conv2DTranspose(ngf * mult,
                                    ngf * mult//2,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias_attr=use_bias),
                BatchNorm2D(ngf * mult//2),
                nn.ReLU()
            ]
            else:
                model += [
                    nn.Conv2DTranspose(ngf * mult,
                                    int(ngf * mult//2),
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias_attr=use_bias),
                    norm_layer(int(ngf * mult // 2)),
                    nn.ReLU()
                ]

        output_size = 2**(6)
        model += [
                nn.Conv2DTranspose(ngf ,
                                output_nc,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias_attr=use_bias),
                nn.Tanh()
                ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)