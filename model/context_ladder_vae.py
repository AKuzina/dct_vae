import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from model.vae import LADDER_VAE, compute_sample_kl
from model.ddgm import DiffusionPrior


class CTX_LADDER_VAE(LADDER_VAE):
    def __init__(self,
                 encoder,
                 decoder,
                 likelihood,
                 beta_start,
                 beta_end,
                 warmup,
                 is_k,
                 latent_scales,
                 free_bits_thr,
                 **kwargs):
        super().__init__(encoder,
                         decoder,
                         likelihood,
                         beta_start,
                         beta_end,
                         warmup,
                         is_k,
                         latent_scales,
                         free_bits_thr)

    def encode(self, batch):
        x = batch[0]
        if self.decoder.ctx_type == 'dct':
            y = batch[-1]
        else:
            y = self.decoder.decoder_blocks[0].x_to_ctx(x)
        encoder_s = self.encoder(x)
        return encoder_s + [y]

    def generate_x(self, N=25, t=None):
        enc_s = [None for _ in range(sum(self.latent_scales)+1)]
        p_xz_params, _, _, _ = self.decoder(enc_s, N=N, t=t)
        p_xz = self.likelihood(*p_xz_params)
        return self.get_x_from_px(p_xz)

    def process_z_L_samples(self, z_L):
        return self.decoder.decoder_blocks[0].ctx_to_x(z_L)