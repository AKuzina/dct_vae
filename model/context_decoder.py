import math
from typing import Optional, Union
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from utils.vae_layers import DecoderResBlock
from datasets.dct import DCT
from utils.distribution import Normal, Delta
from model.ddgm import DiffusionPrior, DiffusionDCTPrior
from utils.thirdparty.unet import UNetModel
from model.decoder import LadderDecoder, quantize


class _CtxDecoderBlock(nn.Module):
    def __init__(self,
                 x_size: list,
                 ctx_size: list,
                 ctx_bits: Union[int, None],
                 ctx_posterior: str,
                 ctx_prior: nn.Module,
                 max_scale: int,
                 next_ch: int,
                 ):
        super().__init__()
        self.__dict__.update(locals())
        self.ctx_prior = ctx_prior
        self.ctx_posterior = self.get_ctx_posterior(ctx_posterior)
        self.posprocess_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=int(max_scale), stride=int(max_scale)),
            nn.Conv2d(self.x_size[0], next_ch, kernel_size=3, padding=1)
        )

    def get_ctx_posterior(self, type, var_init=-10):
        if 'fixed' in type:
            self.ctx_logvar = nn.Parameter(var_init*torch.ones([1] + self.ctx_size), requires_grad=False)
            return lambda x: Normal(x, self.ctx_logvar.repeat(x.shape[0], 1, 1, 1))
        elif 'train' in type:
            self.ctx_logvar = nn.Parameter(var_init*torch.ones([1] + self.ctx_size), requires_grad=True)
            return lambda x: Normal(x, torch.clamp(self.ctx_logvar, -10, 10).repeat(x.shape[0], 1, 1, 1))
        elif 'conditional' in type:
            self.ctx_logvar = nn.Sequential(
                nn.Conv2d(self.ctx_size[0], 100, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(100, self.ctx_size[0], kernel_size=3, padding=1),
                nn.Softplus(0.7),
            )
            return lambda x: Normal(x, torch.clamp(torch.log(self.ctx_logvar(x)), -7, 7))
        elif 'delta' in type:
            return lambda x: Delta(x)
        else:
            raise ValueError(f'unknown ctx posterior: {type}')

    def forward(self, ctx_val, s_dec, mode, t=None):
        """
        :param ctx_val: Analog of s_enc in LadderVAE.
        :param s_dec: Here for compatibility with decoder block interface
        :param mode: train, test
        :param t: temperature
        :return: (p_dist, q_dist, z_sample, s_dec)
        """
        if ctx_val is None:
            q_dist = None
            ctx_val = self.ctx_prior.sample(s_dec.shape[0], t=t)
            ctx_val = quantize(ctx_val, self.ctx_bits)
        else:
            ctx_val = self.preprocess_ctx(ctx_val)
            q_dist = self.ctx_posterior(ctx_val)
            ctx_val = q_dist.sample()

        if isinstance(self.ctx_prior, DiffusionPrior) or isinstance(self.ctx_prior, DiffusionDCTPrior):
            if mode == 'test':
                p_dist = self.ctx_prior.eval_is_ll(ctx_val, is_k=10)
            else:
                p_dist = self.ctx_prior.log_prob(ctx_val, mode=mode, reduce_dim=False)
        else:
            p_dist = self.ctx_prior

        x_ctx_val = self.ctx_to_x(ctx_val)
        x_ctx_val = self.posprocess_block(x_ctx_val)
        s_dec = s_dec + x_ctx_val
        return p_dist, q_dist, ctx_val, s_dec

    def ctx_to_x(self, ctx):
        assert NotImplementedError

    def x_to_ctx(self, x):
        assert NotImplementedError

    def preprocess_ctx(self, ctx):
        """
        In needed,  precprocess context that was created on the dataset construction stage.
        E.g. for DCT context we will do normalization and quantization on this step.
        :param ctx:
        :return:
        """
        return ctx


class DCTDecoderBlock(_CtxDecoderBlock):
    def __init__(self,
                 x_size: list,
                 ctx_size: list,
                 ctx_bits: Union[int, None],
                 ctx_posterior: str,
                 ctx_prior: nn.Module,
                 max_scale: int,
                 next_ch: int,
                 # mode='RGB'
                 ):
        super(DCTDecoderBlock, self).__init__(
            x_size=x_size,
            ctx_size=ctx_size,
            ctx_bits=ctx_bits,
            ctx_posterior=ctx_posterior,
            ctx_prior=ctx_prior,
            max_scale=max_scale,
            next_ch=next_ch,
        )
        self.dct = DCT(x_size[1], x_size[2])

        # DCT scaling parameters
        self.dct_mean = nn.Parameter(torch.zeros(ctx_size), requires_grad=False)
        self.dct_std = nn.Parameter(torch.zeros(ctx_size), requires_grad=False)
        self.dct_scale = nn.Parameter(torch.zeros(ctx_size), requires_grad=False)
        self.std_mult = 4

    def ctx_to_x(self, ctx):
        # unnormalize
        ctx = ctx * self.dct_scale

        # pad with 0 and invert DCT
        pad = self.x_size[1] - ctx.shape[-1]
        x = self.dct.idct2(F.pad(ctx, (0, pad, 0, pad)))
        x = 2 * torch.clamp(x, 0, 1) - 1
        # if self.mode == 'YCbCr':
        #     x = YCBCR_to_RGB(x)
        return x

    def x_to_ctx(self, x, preprocess=True):
        dct = self.dct.dct2(x)[:, :, :self.ctx_size[1], :self.ctx_size[1]]
        if preprocess:
            dct = self.preprocess_ctx(dct)
        return dct

    def preprocess_ctx(self, y_dct):
        # normalize
        y_dct = y_dct / self.dct_scale
        # exactly [-1, 1]
        y_dct = torch.clamp(y_dct, -1, 1)
        y_dct = quantize(y_dct, self.ctx_bits)
        return y_dct


class DownsampleDecoderBlock(_CtxDecoderBlock):
    def __init__(self,
                 x_size: list,
                 ctx_size: list,
                 ctx_bits: Union[int, None],
                 ctx_posterior: str,
                 ctx_prior: nn.Module,
                 max_scale: int,
                 next_ch: int,
                 # mode='RGB'
                 ):
        super(DownsampleDecoderBlock, self).__init__(
            x_size=x_size,
            ctx_size=ctx_size,
            ctx_bits=ctx_bits,
            ctx_posterior=ctx_posterior,
            ctx_prior=ctx_prior,
            max_scale=max_scale,
            next_ch=next_ch,
        )
        self.kernel_size = int(np.ceil(self.x_size[1] / self.ctx_size[1]))
        self.pad_size = int(
            np.ceil((self.kernel_size * self.ctx_size[1] - self.x_size[1]) // 2))

    def ctx_to_x(self, ctx):
        x_up = nn.Upsample(scale_factor=self.kernel_size, mode='bilinear')(ctx)
        up_lim = self.x_size[1] + self.pad_size
        x = x_up[:, :, self.pad_size:up_lim, self.pad_size:up_lim]
        return x

    def x_to_ctx(self, x, preprocess=True):
        x_padded = F.pad(x, (self.pad_size,) * 4)
        x_dwn = nn.AvgPool2d(self.kernel_size)(x_padded)
        if preprocess:
            x_dwn = self.preprocess_ctx(x_dwn)
        return x_dwn

    def preprocess_ctx(self, y_dct):
        y_dct = quantize(y_dct, self.ctx_bits)
        return y_dct


class ContextLadderDecoder(LadderDecoder):
    def __init__(self,
                 num_ch: int,
                 scale_ch_mult: float,
                 block_ch_mult: float,
                 data_ch: int,
                 num_postprocess_blocks: int,
                 likelihood: str,
                 num_mix: int,
                 data_dim: int,
                 weight_norm: bool,
                 batch_norm: bool,
                 latent_scales: list,
                 latent_width: list,
                 num_blocks_per_scale: int,
                 activation: str,
                 arch_mode: str,
                 var_mode: str,
                 softplus_beta: int,
                 z_L_prior: dict,
                 z_L_bits: Union[int, None],
                 disconnect: bool,
                 ctx_type: str,
                 # mode: str,
                 ctx_size: int,
                 ctx_posterior_var: str,
                 ctx_posterior_var_init: float,
                 condition_on_last: bool = False,
                 # add_y_last: bool=False,
                 start_scale_at_x: bool = False,
                 ):
        self.__dict__.update(locals())
        self.max_scale = 2 ** len(latent_scales)
        if self.start_scale_at_x:
            self.max_scale /= 2

        super(ContextLadderDecoder, self).__init__(
            num_ch=num_ch,
            scale_ch_mult=scale_ch_mult,
            block_ch_mult=block_ch_mult,
            data_ch=data_ch,
            num_postprocess_blocks=num_postprocess_blocks,
            likelihood=likelihood,
            num_mix=num_mix,
            data_dim=data_dim,
            weight_norm=weight_norm,
            batch_norm=batch_norm,
            latent_scales=latent_scales,
            latent_width=latent_width,
            num_blocks_per_scale=num_blocks_per_scale,
            activation=activation,
            arch_mode=arch_mode,
            var_mode=var_mode,
            softplus_beta=softplus_beta,
            z_L_prior=None,
            z_L_bits=z_L_bits,
            disconnect=disconnect,
            condition_on_last=condition_on_last,
            start_scale_at_x=start_scale_at_x
        )
        self.init()
        self.z_L_prior = z_L_prior

        # init context decoder block
        self.ctx_size = [data_ch, ctx_size, ctx_size]


        if ctx_type == 'dct':
            ctx_decoder = DCTDecoderBlock(x_size=self.image_size,
                                          ctx_size=self.ctx_size,
                                          ctx_bits=self.z_L_bits,
                                          ctx_posterior=ctx_posterior_var,
                                          ctx_prior=z_L_prior,
                                          max_scale= self.max_scale,
                                          next_ch=self.num_ch[0],
                                          )
        elif ctx_type == 'downsample':
            ctx_decoder = DownsampleDecoderBlock(x_size=self.image_size,
                                          ctx_size=self.ctx_size,
                                          ctx_bits=self.z_L_bits,
                                          ctx_posterior=ctx_posterior_var,
                                          ctx_prior=z_L_prior,
                                          max_scale=self.max_scale,
                                          next_ch=self.num_ch[0],
                                                 )
        # add to the rest of the blocks
        self.decoder_blocks = nn.ModuleList([ctx_decoder, *self.decoder_blocks])

        if not self.disconnect:
            self.y_block =  nn.Sequential(
                nn.AvgPool2d(kernel_size=int(self.max_scale), stride=int(self.max_scale)),
                nn.Conv2d(self.image_size[0], self.num_ch[0], kernel_size=3, padding=1)
            )

        # self.ctx_q = self.get_ctx_posterior(ctx_posterior_var, ctx_posterior_var_init)
        # if self.disconnect or self.condition_on_last: # or self.add_ctx_to_p:
        #     self.z_L_up = self.init_cond_blocks(data_ch, max_scale=self.max_scale)

    def init_cond_blocks(self, cond_width):
        z_L_up = nn.ModuleList()
        scale_sizes = [self.max_scale // (2 ** i) for i in range(self.num_scales)]

        for s_num, (s, w) in enumerate(zip(self.latent_scales, self.latent_width)):
            if s > 0 and s_num > 0:
                z_L_up.append(nn.Sequential(
                    # reshape to the latent's size
                    nn.AvgPool2d(kernel_size=int(scale_sizes[s_num]),
                                 stride=int(scale_sizes[s_num])),
                    # change num channels
                    nn.Conv2d(self.data_ch, self.num_ch[s_num], kernel_size=3, padding=1),
                ))
        return z_L_up

    def z_L_post_proc(self, z_L):
        z_L = self.decoder_blocks[0].ctx_to_x(z_L)
        return z_L

    def init_dct_normalization(self, loader):
        if self.ctx_type == 'dct':
            if hasattr(loader.dataset, 'dataset'):
                self.decoder_blocks[0].dct_mean.data = loader.dataset.dataset.mean
                self.decoder_blocks[0].dct_std.data = loader.dataset.dataset.std
                self.decoder_blocks[0].dct_scale.data = loader.dataset.dataset.scale
            else:
                self.decoder_blocks[0].dct_mean.data = loader.dataset.mean
                self.decoder_blocks[0].dct_std.data = loader.dataset.std
                self.decoder_blocks[0].dct_scale.data = loader.dataset.scale

            if isinstance(self.decoder_blocks[0].ctx_prior, DiffusionPrior):
                S = self.decoder_blocks[0].dct_scale.data
                if self.decoder_blocks[0].ctx_prior.use_noise_scale:
                    self.decoder_blocks[0].ctx_prior.noise_scale = nn.Parameter(S, requires_grad=False)
