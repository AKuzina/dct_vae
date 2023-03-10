import math
from typing import Optional, Union
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from hydra.utils import instantiate

from utils.vae_layers import DecoderResBlock
from datasets.dct import DCT, YCBCR_to_RGB
from utils.distribution import Normal, Delta
from model.ddgm import DiffusionPrior, DiffusionDCTPrior
from utils.thirdparty.unet import UNetModel
from model.plain_decoder import _Decoder, PlainDecoder


class _DecoderBlock(nn.Module):
    """
    Single block of the Deep Hierarchical VAE decoder

    Input:   s_enc - features from the encoder, from the same level
             s_dec - feature from the previous decoder level

    Output:  p(z_i | ... )   - prior for the current latent
             q(z_i | x, ...) - variational posterior for the current latent
    """
    def __init__(self,
                 in_channels: int,
                 ch_mult: float,
                 num_blocks_per_scale: int,
                 p_width: int,
                 out_channels: int,
                 z_width: int,
                 upsample: Union[int, None],
                 conv_block_params: dict,
                 var_mode: str,
                 softplus_beta: Union[float, None],
                 top_prior: Union[nn.Module, None] = None,
                 ):
        super().__init__()
        self.__dict__.update(locals())
        # how to model variance (in both q and p)
        assert var_mode in ['log', 'softplus'], 'Unknown variance type'
        self.z_up = nn.Conv2d(z_width, in_channels, 1, padding=0)
        # Define backbone NN
        self.resnet = nn.Sequential(*[
            DecoderResBlock(in_channels,
                            int(in_channels * ch_mult),
                            in_channels if b_num + 1 < num_blocks_per_scale else out_channels,
                            stride=1,
                            mode='2x3',
                            use_res=True,
                            zero_last=False,
                            **conv_block_params)
            for b_num in range(num_blocks_per_scale)
        ])
        # Upsample output f the block (if required)
        self.upsample = nn.Upsample(size=upsample, mode='nearest') \
            if upsample is not None else nn.Identity()

        # define NN to get parameters of the prior
        self.top_prior = top_prior
        self.init_p_net()

    def init_p_net(self):
        if self.top_prior is None:
            self.s_to_p = DecoderResBlock(self.in_channels,
                                      int(self.in_channels*self.ch_mult),
                                      self.p_width,
                                      stride=1,
                                      use_res=False,
                                      zero_last=True,
                                      **self.conv_block_params)
    def get_logvar(self, lv):
        """
        Given the output of the NN, returns the log-variance of the distribution
        :param lv: output of the NN for the var
        :return:
        """
        if self.var_mode == 'softplus':
            sp = nn.Softplus(self.softplus_beta)
            lv = torch.log(sp(lv))
        return torch.clamp(lv, -5, 0)

    def get_q_p_dist(self, s_enc, s_dec, mode):
        raise NotImplementedError

    def forward(self, s_enc, s_dec, mode, t=None):
        """
        :param s_enc: [MB, z_width, scale, scale]
        :param s_dec: (s_{i+1}) [MB, in_ch, sc, sc]
        :param mode: train or test
        :return: z sample q an p distributions, deterministic features (s_out) to pass to the next block
        """
        p_dist, q_dist, s_dec = self.get_q_p_dist(s_enc, s_dec, mode)
        assert mode in ['train', 'val', 'test']
        # if mode == 'decode':
        if s_enc is not None:  # -> decoding
            z = q_dist.sample(t=t)
        else:  # -> sampling
            if self.top_prior is None:
                z = p_dist.sample(t=t)
            else:
                N = s_dec.shape[0]
                z = p_dist.sample(N, t=t)
        s_dec = self.upsample(self.resnet(s_dec + self.z_up(z)))

        if isinstance(p_dist, DiffusionPrior):
            if mode == 'test':
                p_dist = p_dist.eval_is_ll(z, is_k=2)
            else:
                p_dist = p_dist.log_prob(z, mode=mode)
        return p_dist, q_dist, z, s_dec


class DecoderBlock(_DecoderBlock):
    def __init__(self,
                 in_channels: int,
                 z_width: int,
                 ch_mult: float,
                 out_channels: int,
                 num_blocks_per_scale: int,
                 conv_block_params: dict,
                 upsample: Union[int, None],
                 var_mode: str,
                 softplus_beta: Union[float, None],
                 disconnect: bool=False,
                 top_prior: Union[nn.Module, None] = None,
                 ):
        """
                   -------------- s_dec---------------
                   ↓                |                ↓
          s_enc -→ q                |              p, h
                                    ↓
                        z ~ q if decoder else p
                                    ↓
                              z + s_dec + h
                                    ↓ (resnet)
                                  s_out

        Implements the decoder block from the vdvae paper
        s_enc and s_dec are inputs from the previous blocks (encoder and decoder correspondingly)
        """
        super(DecoderBlock, self).__init__(
            in_channels=in_channels,
            ch_mult=ch_mult,
            num_blocks_per_scale=num_blocks_per_scale,
            p_width=2 * z_width + in_channels,
            out_channels=out_channels,
            z_width=z_width,
            upsample=upsample,
            conv_block_params=conv_block_params,
            var_mode=var_mode,
            softplus_beta=softplus_beta,
            top_prior=top_prior,
        )
        # if disconnect is True, q will only depend on the s_enc
        self.disconnect = disconnect
        self.s_to_q = DecoderResBlock(in_channels,
                                      max(int(in_channels*ch_mult), 1),
                                      2*z_width,
                                      stride=1,
                                      mode='2x3',
                                      use_res=False,
                                      zero_last=False,
                                      **conv_block_params)
        self.init_q()

    def init_q(self):
        i = -2
        if isinstance(self.s_to_q.net[-1], nn.Conv2d):
            i = -1
        nn.init.uniform_(self.s_to_q.net[i].weight, -1, 1)

    def get_q_p_dist(self, s_enc, s_dec, mode):
        if self.top_prior is None:
            # get parameters of the prior
            p_out = self.s_to_p(s_dec)
            p_params, h = p_out[:, :2 * self.z_width], p_out[:, 2 * self.z_width:]
            p_mu, p_logvar = torch.chunk(p_params, 2, dim=1)
            p_logvar = self.get_logvar(p_logvar)
            # if t is not None:
            #     p_logvar += torch.ones_like(p_logvar) * math.log(t)
            p_dist = Normal(p_mu, p_logvar)
        else:
            p_dist = self.top_prior
            if isinstance(p_dist, Normal):
                with torch.no_grad():
                    p_dist.log_var.clamp_(-4, 0)

        # get parameters of the variational posterior
        if s_enc is not None:
            if self.disconnect:
                s = s_enc
            else:
                s = s_enc + s_dec
            q_mu, q_logvar = torch.chunk(self.s_to_q(s), 2, dim=1)
            if isinstance(p_dist, DiffusionPrior):
                q_mu = torch.tanh(q_mu)
            q_logvar = self.get_logvar(q_logvar)
            q_dist = Normal(q_mu, q_logvar)
        else:
            q_dist = None
        # add prior features to the output
        if self.top_prior is None:
            s_dec = s_dec + h
        return p_dist, q_dist, s_dec


class LadderDecoder(_Decoder):
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
                 condition_on_last: bool = False,
                 start_scale_at_x: bool = False,
                 ):
        super(LadderDecoder, self).__init__()
        self.__dict__.update(locals())
        assert len(latent_width) == len(latent_scales)

        self.conv_block_params = {
            'batch_norm': batch_norm,
            'weight_norm': weight_norm,
            'activation': self.get_activation(activation),
        }

        self._decoder_block = lambda args: {
            # 'delta': DecoderBlockDelta,
            'separate': DecoderBlock
        }[arch_mode](**args,
                     ch_mult=block_ch_mult,
                     num_blocks_per_scale=num_blocks_per_scale,
                     conv_block_params=self.conv_block_params,
                     var_mode=var_mode,
                     softplus_beta=softplus_beta,
                     disconnect=disconnect,
                     )
        self.num_scales = len(latent_scales)
        self.num_latents = sum(latent_scales)
        self.image_size = [data_ch, data_dim, data_dim]
        self.num_ch = [num_ch]
        for i in range(self.num_scales-1):
            self.num_ch += [int(self.num_ch[-1] * scale_ch_mult)]
        # reverse the order of latents: from top (z_L) to bottom (z_1)
        self.num_ch.reverse()
        self.latent_scales.reverse()
        self.latent_width.reverse()
        print('Decoder channels', self.num_ch)
        self.num_p_param = _Decoder.get_num_lik_params(likelihood)

        # create dummy s_L input
        L_dim = data_dim // (2 ** self.num_scales)
        if start_scale_at_x:
            L_dim = (2 * data_dim) // (2 ** self.num_scales)
        self.s_L = nn.Parameter(torch.zeros(1, self.num_ch[0], L_dim, L_dim),
                                requires_grad=False)

        # init the NNs
        if isinstance(z_L_prior, Normal):
            z_L_prior = None
        self.decoder_blocks = self.init_decoder_blocks(z_L_prior)
        self.post_process = self.init_post_process()
        if self.disconnect or self.condition_on_last:
            # if we use "disconnected" decoder we need conditioning on z_L for each latent
            # init the NN which will reshape z_L to the size of s_enc (they will be summed up)
            self.z_L_up = self.init_cond_blocks(self.latent_width[0])
        self.init()

    def forward(self,
                encoder_s: list,
                N: Optional[int] = None,
                t: Optional[float] = None,
                mode: str = 'train',
                ):
        """
        Decoder of the ladder VAE
        :param encoder_s: list of deterministic features from encoder
                           in the bottom-up order [q_1, ..., q_L]
        :param N: number of samples from p(z_L)
        :param t: temperature
        :return: tuple(p_xz_parameters, p_dist, q_dist, z_samples):
            parameters of the conditional generative distribution p(x|{z}),
            list of prior distributions
            list of posterior distributions (or None)
            list of latent variables z
        """
        encoder_s.reverse()  # [s_enc_L, s_enc_{L-1}, ..., s_enc_1]
        # init s_dec
        if encoder_s[0] is not None:  # -> reconstruction
            N = encoder_s[0].shape[0]
        s_dec = self.s_L.repeat(N, 1, 1, 1)

        p_dist = []
        q_dist = []
        z_list = []
        scale, s = s_dec.shape[-1], 0
        N_blocks = len(self.decoder_blocks)
        for i, dec_block in enumerate(self.decoder_blocks):
            s_enc = encoder_s[i]
            if i > 0:
                if self.disconnect and s_enc is not None:
                    s_enc = s_enc + self.z_L_up[i](z_L)
            # print('s_enc', s_enc.shape, 's_dec', s_dec.shape)
            p, q, z, s_dec = dec_block(s_enc, s_dec, mode, t=t)
            p_dist.append(p)
            q_dist.append(q)
            z_list.append(z)
            if i == 0:
                z_L = self.z_L_post_proc(z_list[0])
            if self.condition_on_last:
                if scale != s_dec.shape[-1] and  i < (N_blocks - 1):
                    s_dec = s_dec + self.z_L_up[s](z_L)
                    s += 1
                    scale = s_dec.shape[-1]

        if self.disconnect:
            s_dec = s_dec + self.z_L_up[-1](z_L)
        p_xz_params = self.get_p_xz_params(self.post_process(s_dec))
        return p_xz_params, p_dist, q_dist, z_list

    def forward_sample(self, N: int = 1, freq: int =1):
        # init s_dec
        s_dec = self.s_L
        s_saved = [s_dec]
        s_ind = [0]
        z_L = None
        scale, s = self.s_L.shape[-1], 0
        N_blocks = len(self.decoder_blocks)
        for i, dec_block in enumerate(self.decoder_blocks):
            p, q, z, s_dec = dec_block(None, s_dec, 'train')
            if i == 0:
                z_L = self.z_L_post_proc(z)
            if self.condition_on_last:
                if scale != s_dec.shape[-1] and  i < (N_blocks - 1):
                    s_dec = s_dec + self.z_L_up[s](z_L)
                    s += 1
                    scale = s_dec.shape[-1]
            s_ind.append(s)
            s_saved.append(s_dec)

        rows = []
        z_L = z_L.data.repeat(N, 1, 1, 1)
        for n_fixed in range(1, len(s_saved), freq):
            s_dec = s_saved[n_fixed].repeat(N, 1, 1, 1)
            scale = s_dec.shape[-1]
            s = s_ind[n_fixed]
            for j in range(n_fixed, len(self.decoder_blocks)):
                p, q, z, s_dec = self.decoder_blocks[j](None, s_dec, 'train')
                if self.condition_on_last:
                    if scale != s_dec.shape[-1] and  i < (N_blocks - 1):
                        s_dec = s_dec + self.z_L_up[s](z_L)
                        s += 1
                        scale = s_dec.shape[-1]
            if self.disconnect:
                s_dec = s_dec + self.z_L_up[-1](z_L)
            p_xz_params = self.get_p_xz_params(self.post_process(s_dec))
            rows.append(p_xz_params)
        return z_L,  rows

    def init_cond_blocks(self, cond_width):
        z_L_up = nn.ModuleList()
        z_L_dim = self.s_L.shape[-1]
        scale_sizes = [z_L_dim * (2 ** i) for i in range(self.num_scales+1)]
        # for s_num in range(self.num_scales):
        for s_num, (s, w) in enumerate(zip(self.latent_scales, self.latent_width)):
            if s > 0 and s_num > 0:
                # reshape to the latent's size and change num channels
                z_L_up.append(nn.Sequential(
                    nn.Upsample(size=scale_sizes[s_num], mode='nearest'),
                    nn.Conv2d(cond_width, self.num_ch[s_num], kernel_size=1, padding=0),
                ))
        return z_L_up

    def z_L_post_proc(self, z_L):
        return z_L

    def init_decoder_blocks(self, top_prior) -> nn.ModuleList:
        decoder_backbone = nn.ModuleList()
        # S_L = self.image_size[1] // (2 ** self.num_scales)
        z_L_dim = self.s_L.shape[-1]
        scale_sizes = [z_L_dim * (2 ** i) for i in range(1, self.num_scales + 1)]
        scale_sizes[-1] = self.image_size[1]
        for i in range(self.num_scales)[::-1]:
            if self.latent_scales[i] == 0:
                scale_sizes[i-1] = scale_sizes[i]
        for s_num, (s, w) in enumerate(zip(self.latent_scales, self.latent_width)):
            ss = scale_sizes[s_num-1] if s_num > 0 else z_L_dim
            print(f'Scale {self.num_scales-s_num}, {s} latents, out shape: {int(ss)}')

            for latent in range(s):
                out_ch = self.num_ch[s_num]
                is_last = latent+1 == s
                if is_last and s_num+1 < len(self.latent_scales):
                    out_ch = self.num_ch[s_num+1]
                block_params = {
                    'in_channels': self.num_ch[s_num],
                    'z_width': w,
                    'out_channels': out_ch,
                    'upsample': scale_sizes[s_num] if is_last else None,
                    'top_prior': top_prior if (s_num + latent) == 0 else None,
                }
                decoder_backbone.append(self._decoder_block(block_params))

                # stable init for the resnet
                res_nn = decoder_backbone[-1].resnet[-1]
                res_nn.net[res_nn.last_conv_id].weight.data *= math.sqrt(1. / self.num_latents)
                decoder_backbone[-1].z_up.weight.data *= math.sqrt(1. / self.num_latents)
        return decoder_backbone

    def init_post_process(self) -> nn.Sequential:
        act_out = nn.Sigmoid() if self.num_p_param == 1 else nn.Identity()
        post_net = []
        for i in range(self.num_postprocess_blocks):
            is_last = i+1 == self.num_postprocess_blocks
            post_net.append(
                DecoderResBlock(self.num_ch[-1],
                                int(self.num_ch[-1]*self.block_ch_mult),
                                self.num_ch[-1],
                                stride=1,
                                mode='2x3',
                                use_res=True,
                                zero_last=True if is_last else False,
                                **self.conv_block_params)
            )
        out_ch = self.num_p_param * self.image_size[0]
        if self.likelihood == 'logistic_mixture':
            out_ch = self.num_mix * (out_ch + 1)
        post_net += [
            nn.Conv2d(self.num_ch[-1], out_ch, kernel_size=3, padding=1),
            act_out
        ]
        post_net[-2].bias.data *= 0.
        return nn.Sequential(*post_net)

    def get_p_xz_params(self, out_feature) -> tuple:
        if self.likelihood == 'logistic_mixture':
            log_probs, ll = out_feature[:, :self.num_mix], out_feature[:, self.num_mix:]
            ll = ll.reshape(-1, self.image_size[0], self.num_mix*self.num_p_param, self.image_size[1], self.image_size[2])
            return (log_probs, ) + torch.chunk(ll, self.num_p_param, dim=2)
        else:
            return torch.chunk(out_feature, self.num_p_param, dim=1)


def quantize(x, n_bits=6):
    x = (x + 1) / 2.  # [-1, 1] -> [0, 1]
    x[x >= 1.] = 0.999
    noise = 0.25 * (torch.rand_like(x)*2 - 1)
    # noise = torch.zeros_like(x)
    x = torch.floor(x * 2. ** n_bits + noise) + 0.5 #[0, 1] -> [0, 2^bits]
    x = x / 2. ** n_bits
    x = x * 2. - 1  # back to [-1, 1]
    return x

def dequantize(x, n_bits=6):
    eps = (torch.rand_like(x) * 2 - 1) / (2 ** n_bits)
    return x + eps