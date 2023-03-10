import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# from utils.flow_layers import AffineCoupling1d, AffineCoupling2d
from utils.nn import Siren
# from utils.arm_layers import CausalConv1d, GatedResidualLayer


class Distribution(nn.Module):
    def __init__(self):
        super(Distribution, self).__init__()

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, N=1, t=None):
        raise NotImplementedError


class Normal(Distribution):
    def __init__(self, mu, log_var, *args, **kwargs):
        super(Normal, self).__init__()
        self.mu = mu
        self.log_var = log_var

    def log_prob(self, x, reduce_dim=True):
        MB = x.shape[0]
        if len(x.shape) > len(self.mu.shape):
            MB = x.shape[0]
        log_p = -0.5 * (math.log(2.0*math.pi) +
                        self.log_var +
                        torch.pow(x - self.mu, 2) / (torch.exp(self.log_var) + 1e-10))
        if reduce_dim:
            return log_p.reshape(MB, -1).sum(1)
        else:
            return log_p.reshape(MB, -1)

    def sample(self, N=None, t=None):
        size = self.mu.shape
        if N is not None:
            size = torch.Size([N]) + size
        z_sample = torch.empty(size, device=self.mu.device)
        if t is not None:
            sigma = (0.5 * (self.log_var + torch.ones_like(self.log_var) * math.log(t))).exp()
        else:
            sigma = (0.5*self.log_var).exp()
        eps = z_sample.normal_()
        return self.mu + sigma*eps

    def update(self, delta_mu, delta_logvar):
        self.mu = self.mu + delta_mu
        self.log_var = self.log_var + delta_logvar

    def get_E(self):
        return self.mu

    def entropy(self):
        c = 1 + math.log(math.pi*2)
        return 0.5 * (c + self.log_var).sum()

    def kl(self, dist):
        """
        compute kl-divergence with the given distribution
        """
        assert isinstance(dist, Normal), 'Can only compute analytical kl for gaussians'
        log_v_r = dist.log_var - self.log_var
        mu_r_sq = (self.mu - dist.mu) ** 2
        kl = 0.5 * (-1 + log_v_r + (self.log_var.exp() + mu_r_sq) / dist.log_var.exp())
        return kl


def create_standard_normal_prior(size):
    size = list(size)
    mu = nn.Parameter(torch.zeros(size), requires_grad=False)
    logvar = nn.Parameter(torch.zeros(size), requires_grad=False)
    return Normal(mu, logvar)


def create_gaussian_prior(size):
    size = list(size)
    mu = nn.Parameter(torch.zeros(size), requires_grad=True)
    logvar = nn.Parameter(torch.randn(size)*0.01, requires_grad=True)
    return Normal(mu, logvar)


class Delta(Distribution):
    def __init__(self, x):
        self.x = x

    def log_prob(self, x, reduce_dim=True):
        out =  torch.zeros(x.shape, device=x.device).reshape(x.shape[0], -1)
        if reduce_dim:
            out = out.sum(1)
        return out

    def sample(self, N=None):
        x_sample = self.x.clone()
        if N is not None:
            size = torch.Size([N]) + self.x.size()
            x_sample = x_sample.unsqueeze(0).repeate(size)
        return x_sample

    def get_E(self):
        return self.x


class Bernoulli(Distribution):
    def __init__(self, p, *args, **kwargs):
        super(Bernoulli, self).__init__()
        self.p = torch.clamp(p, min=1e-7, max=1.-1e-7)

    def log_prob(self, x):
        MB = x.shape[0]
        assert torch.max(x).item() <= 1.0 and torch.min(x).item() >= 0.0
        log_p = x * torch.log(self.p) + (1. - x) * torch.log(1. - self.p)
        return log_p.reshape(MB, -1).sum(1)

    def sample(self, N=None):
        p = self.p
        if N is not None:
            p = p.unsqueeze(0).repeat([N] + [1 for _ in range(len(p.shape))])
        return torch.bernoulli(p)

    def get_E(self):
        return self.p


class Logistic256(Distribution):
    def __init__(self, mean, var, *args, **kwargs):
        super(Logistic256, self).__init__()
        self.mean = mean
        softplus = nn.Softplus(0.4)
        self.log_var = torch.log(softplus(torch.clamp(var, min=-20.)))

    def log_prob(self, x, low_bit=False):
        assert x.min() >= -1. and x.max() <= 1.
        # rescale x to [-1, 1] if needed
        if x.min() >= 0:
            x = 2. * x - 1

        if low_bit:
            max_bit = 31.
        else:
            max_bit = 255.

        centered = x - self.mean  # B, C, H, W
        inv_stdv = torch.exp(- self.log_var)

        # each pixel has a bin of width 2/n_bit -> half of the bin is 1/n_bit
        plus_in = inv_stdv * (centered + 1. / max_bit)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_stdv * (centered - 1. / max_bit)
        cdf_min = torch.sigmoid(min_in)

        # probability to be in the bin
        # cdf_delta = cdf_plus - cdf_min
        cdf_delta = torch.clamp(cdf_plus - cdf_min, min=1e-10)
        log_probs = torch.log(cdf_delta)

        # for pixel 0 we have -\inf instead of min_in
        log_cdf_plus = plus_in - F.softplus(plus_in)
        pix_0 = -1. + 1./max_bit
        log_probs = torch.where(x <= pix_0,
                                log_cdf_plus,
                                log_probs)

        # for pixel 255 we have \inf instead of plus_in
        log_one_minus_cdf_min = -F.softplus(min_in)
        pix_255 = 1. - 1./max_bit
        log_probs = torch.where(x >= pix_255,
                                log_one_minus_cdf_min,
                                log_probs)
        log_probs = log_probs.sum(dim=[1, 2, 3])  # MB
        return log_probs

    def sample(self, N=None, t=None):
        size = self.mean.shape
        if N is not None:
            size = torch.Size([N]) + size
        u = torch.Tensor(size).uniform_(1e-5, 1. - 1e-5)
        u = u.to(self.mean.device)
        if t is not None:
            scale = torch.exp(self.log_var + torch.ones_like(self.log_var) * math.log(t))
        else:
            scale = torch.exp(self.log_var)
        x = self.mean + scale * (torch.log(u) - torch.log(1. - u))
        return x

    def get_E(self):
        return self.mean

    def entropy(self):
        return self.logvar + 2


class MixtureLogistic256(Distribution):
    # Using the implementations from
    # https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py
    # https://github.com/openai/vdvae/blob/ea35b490313bc33e7f8ac63dd8132f3cc1a729b4/vae_helpers.py
    def __init__(self, logit_probs, mean, log_var): #, coeffs):
        super(MixtureLogistic256, self).__init__()
        self.logit_probs = logit_probs  # MB, M, H, W
        self.data_ch = 3
        mb, self.num_mix, h, w = logit_probs.shape
        self.means = mean  # MB, 3, M, H, W
        softplus = nn.Softplus(0.4)
        self.log_var = torch.log(softplus(torch.clamp(log_var, min=-20.))) # MB, 3, M, H, W

    def log_prob(self, x, low_bit=False):

        assert x.min() >= -1. and x.max() <= 1.
        # rescale x to [-1, 1] if needed
        if x.min() >= 0:
            x = 2. * x - 1

        if low_bit:
            max_bit = 31.
        else:
            max_bit = 255.

        x = x.unsqueeze(2)  # MB, 3, 1, H, W
        centered_x = x - self.means  # B, C, M, H, W

        inv_stdv = torch.exp(-self.log_var)

        # each pixel has a bin of width 2/n_bit -> half of the bin is 1/n_bit
        plus_in = inv_stdv * (centered_x + 1. / max_bit)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_stdv * (centered_x - 1. / max_bit)
        cdf_min = torch.sigmoid(min_in)

        # probability to be in the bin
        cdf_delta = torch.clamp(cdf_plus - cdf_min, min=1e-10)
        log_probs = torch.log(cdf_delta)

        # for pixel 0 we have -\inf instead of min_in
        log_cdf_plus = plus_in - F.softplus(plus_in)
        pix_0 = -1. + 1./max_bit
        log_probs = torch.where(x.repeat(1, 1, self.num_mix, 1, 1) <= pix_0,
                                log_cdf_plus,
                                log_probs)

        # for pixel 255 we have \inf instead of plus_in
        log_one_minus_cdf_min = -F.softplus(min_in)
        pix_255 = 1. - 1./max_bit
        log_probs = torch.where(x.repeat(1, 1, self.num_mix, 1, 1) >= pix_255,
                                log_one_minus_cdf_min,
                                log_probs)

        # MB x M x H x W
        log_probs = torch.sum(log_probs, dim=1) + F.log_softmax(self.logit_probs, dim=1)
        # now get rid of the mixtures with log sum exp
        log_probs = torch.logsumexp(log_probs, 1)  # MB x H x W
        log_probs = log_probs.sum(dim=[1, 2])  # MB
        return log_probs

    def sample(self, t=None):
        # sample mixture num
        eps = torch.empty_like(self.logit_probs).uniform_(1e-5, 1. - 1e-5)  # MB, M, H, W
        amax = torch.argmax(self.logit_probs - torch.log(-torch.log(eps)), dim=1)
        sel = one_hot(amax, self.logit_probs.size()[1], dim=1, device=self.means.device).unsqueeze(1) # MB, 1, M, H, W

        # select logistic parameters -> MB, 3, H, W
        means = (self.means * sel).sum(2)
        log_scales = (self.log_var * sel).sum(2)
        if t is not None:
            log_scales = log_scales + torch.ones_like(self.log_scales) * math.log(t)

        # sample from logistic & clip to interval
        u = torch.empty_like(means).uniform_(1e-5, 1. - 1e-5)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
        return torch.clamp(x, -1, 1)

    def get_E(self):
        raise NotImplementedError


def one_hot(indices, depth, dim, device):
    """
    https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/cfecc7b1776b85d09d9336f07a6b886c3ca8e486/efficient_vdvae_torch/utils/utils.py#L43
    """
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=device)
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot


