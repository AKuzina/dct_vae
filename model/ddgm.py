import torch
import torch.nn as nn
import numpy as np
import math

from utils.distribution import Normal


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class DiffusionPrior(nn.Module):
    def __init__(self,
                 model,
                 T,
                 beta_schedule,
                 t_sample='uniform',
                 parametrization='x',
                 num_bits=5,
                 ll='discretized_gaussian',
                 use_noise_scale=False,
                 ):
        super(DiffusionPrior, self).__init__()
        # A NeuralNet (unet), which takes as input z and the timestamp
        self.parametrization = parametrization
        self.num_bits = num_bits
        self.ll = ll

        self.use_noise_scale = use_noise_scale
        if self.use_noise_scale:
            shape = [model.in_channels, model.image_size, model.image_size]
            self.noise_scale = nn.Parameter(torch.zeros(shape), requires_grad=True)

        assert parametrization in ['x', 'eps', 'x_var'], \
            f'unknown parametrization {parametrization}. Expect to be x0 or eps'
        self.model = model
        self.device = None
        # num stesps
        self.T = T
        # how to sample t during training: uniform or loss-aware
        self.t_sample = t_sample
        self._ll_hist = -1.*torch.ones(self.T)
        # create beta schedule
        self.beta_schedule = beta_schedule
        self.beta = self.get_beta_schedule(beta_schedule)
        self.alphas_cumprod = np.cumprod(1.0 - self.beta, axis=0)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.beta * (1.0 - np.append(1.0, self.alphas_cumprod[:-1])) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                self.beta * np.sqrt(np.append(1.0, self.alphas_cumprod[:-1])) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - np.append(1.0, self.alphas_cumprod[:-1]))
                * np.sqrt(1 - self.beta)
                / (1.0 - self.alphas_cumprod)
        )
        # variance of the p distribution (fixed)
        # self.p_log_variance = self.posterior_log_variance_clipped

    def get_beta_schedule(self, name):
        s = 0.008
        if name == "cosine":
            betas = []
            max_beta = 0.999
            fn = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            for i in range(self.T):
                t1 = i / self.T
                t2 = (i + 1) / self.T
                betas.append(min(1 - fn(t2) / fn(t1), max_beta))
            return np.array(betas)
        elif name == 'linear':
            if self.T < 21:
                scale = 100 / self.T
                multiply = 0.01
            else:
                scale = 1000 / self.T
                multiply = 0.001
            beta_start = scale * multiply
            beta_end = scale * 0.02
            return np.linspace(beta_start, beta_end, self.T, dtype=np.float64)
        else:
            raise NotImplementedError(f"unknown beta schedule: {name}")

    def forward(self, x):
        pass

    def sample(self, N, t=1.):
        '''
        t stand for temperature.
        '''
        shape = [N, self.model.in_channels,  self.model.image_size, self.model.image_size]
        img = torch.randn(*shape, device=self.device)
        indices = list(range(self.T))[::-1]
        for i in indices:
            t_step = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                img = self.p_sample(img, t_step, temp=t)
        if self.use_noise_scale:
            img = img / self.noise_scale
        return img

    def q_sample(self, z_0, t):
        noise = torch.randn_like(z_0)
        out = (
                _extract_into_tensor(np.sqrt(self.alphas_cumprod), t, z_0.shape) * z_0
                + _extract_into_tensor(np.sqrt(1.0 - self.alphas_cumprod), t, z_0.shape)
                * noise
        )
        return out

    def p_sample(self, z_t, t, temp=1.):
        '''
        Sample from p(z_{t-1}|z_t)
        '''
        p_dist = self.get_p(z_t, t)
        p_sample = p_dist.sample(t=temp)

        # no sampling for the step 0
        p_sample[t == 0] = p_dist.mu[t==0]
        return p_sample

    def get_p(self, z_t, t):
        if self.parametrization == 'x':
            p_mean, p_logvar_coef = torch.chunk(self.model(z_t, t), 2, dim=1)
            if self.num_bits is None:
                p_logvar_coef = torch.clamp(p_logvar_coef, -5, 0)
            p_dist = Normal(p_mean, p_logvar_coef)
        elif self.parametrization == 'x_var':
            p_mean, p_logvar_coef = torch.chunk(self.model(z_t, t), 2, dim=1)
            # p_logvar_coef = torch.tanh(p_logvar_coef)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, z_t.shape
            )
            p_dist = Normal(p_mean, min_log)
        elif self.parametrization == 'eps':
            eps = self.model(z_t, t)
            z0_pred = self._predict_z0_from_eps(z_t, t, eps)
            p_dist = self.get_q_posterior(z0_pred, z_t, t)
        return p_dist

    def get_q_posterior(self, z_0, z_t, t):
        q_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, z_t.shape) * z_0
                + _extract_into_tensor(self.posterior_mean_coef2, t, z_t.shape) * z_t
        )
        q_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, z_t.shape
        )
        return Normal(q_mean, q_log_variance_clipped)

    def sample_t(self, batch_size, mode):
        if self.t_sample == 'uniform' or mode == 'val':
            indices_np = np.random.choice(range(self.T), size=(batch_size,))
        elif self.t_sample == 'loss_aware':
            # sample with the weights proportional to the loss
            loss = -1 * self._ll_hist.numpy()
            weights = loss/np.sum(loss)
            indices_np = np.random.choice(range(self.T), size=(batch_size,), p=weights)
        else:
            NotImplementedError(f"unknown t sampling schedule: {self.t_sample}")
        return indices_np

    def eval_is_ll(self, z_0, is_k=1):
        """
        Importance sampling estimation of the NLL
        :param z_0: batch of data points
        :param is_k: number of importance samples
        :return:
        """

        elbo = torch.zeros(is_k, z_0.shape[0], device=z_0.device)
        for k in range(is_k):
            elbo[k] = self.log_prob(z_0, mode='test')
        ll = torch.logsumexp(elbo, 0) - np.log(is_k)
        return ll

    def log_prob(self, z_0, mode='train', reduce_dim=True):
        """
        :param z_0: (MB, ch, h, w)
        :param mode: 'train', 'test' or 'val'
        :return:
        """
        batch_size = z_0.shape[0]
        if self.device is None:
            self.device = z_0.device
        # loop over t or sample t uniformly
        if mode in ['test', 'val']:
            t_to_loop = [torch.ones(batch_size, ).long().to(z_0.device)*i for i in range(self.T)]
        else:
            indices_np = self.sample_t(batch_size, mode=mode)
            t_to_loop = [
                torch.from_numpy(indices_np).long().to(z_0.device)
            ]
        log_lik = 0.
        for t in t_to_loop:
            log_lik += self._step_loss(z_0, t, reduce_dim=reduce_dim)

        if mode == 'val' and self.t_sample == 'loss_aware':
            t = t_to_loop[0]
            with torch.no_grad():
                # update running average of the loss for the sampler
                for i in range(self.T):
                    step_ll = torch.clamp(log_lik[t == i], max=0)
                    if len(step_ll) > 0:
                        gamma = 0.99
                        self._ll_hist[i] = gamma*self._ll_hist[i] + (1 - gamma)*step_ll.mean().cpu()
        # multiply by the number of steps if sampling is used
        log_lik *= self.T / len(t_to_loop)
        return log_lik

    def _step_loss(self, z_0, t, reduce_dim=True):
        batch_size = z_0.shape[0]
        # sample z_t from q(z_t | z_0)
        if self.use_noise_scale:
            z_0 = z_0 * self.noise_scale
            # z_0 = z_0 / self.noise_scale.abs().mean()
        z_t = self.q_sample(z_0, t)
        # get p(z_{t-1} | z_t) params
        p_dist = self.get_p(z_t, t)
        # get q posterior params

        q_posterior_dist = self.get_q_posterior(z_0, z_t, t)
        if self.ll == 'discretized_gaussian':
            rec_ll = discretized_gaussian_log_likelihood(z_0, means=p_dist.mu,
                                                         log_scales=0.5 * p_dist.log_var,
                                                         n_bits=self.num_bits)
            rec_ll = rec_ll.reshape(batch_size, -1)
            if reduce_dim:
                rec_ll = rec_ll.sum(1)
        elif 'vdm':
            vocab_size = 2 ** self.num_bits
            exp_half_g_0 = _extract_into_tensor(np.sqrt( (1 - self.alphas_cumprod) / self.alphas_cumprod), torch.zeros_like(t), z_0.shape)
            # var_0 = torch.sigmoid(g_0)
            eps_0 = torch.rand_like(z_0)
            z_noisy_rescaled = z_0 + exp_half_g_0 * eps_0  # = z_0/sqrt(1-var)
            x = (torch.clamp(z_0, -1., 1.) + 1.)/ 2. * (vocab_size - 1) # covert to uint

            # create x OHE
            x = x.round().long()
            x_onehot = torch.nn.functional.one_hot(x, num_classes=vocab_size)

            # decoding
            x_vals = torch.arange(0, vocab_size)[:, None]
            x_vals = x_vals.repeat(1, self.model.in_channels)
            x_vals = 2 * ((x_vals + .5) / vocab_size) - 1
            x_vals = x_vals.transpose(1, 0)[None, None, None, :, :]
            x_vals = x_vals.swapaxes(2, 3).swapaxes(1, 2)
            x_vals = x_vals.to(z_0.device)
            inv_stdev = 1 / exp_half_g_0[..., None] # torch.exp(-0.5 * g_0[..., None])
            logits = -0.5 * torch.square((z_noisy_rescaled[..., None] - x_vals) * inv_stdev)
            # calcualte log prob
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            rec_ll = torch.sum(x_onehot * logprobs, axis=-1).reshape(batch_size, -1)
            if reduce_dim:
                rec_ll = rec_ll.sum(1)

        kl = q_posterior_dist.kl(p_dist).reshape(batch_size, -1)
        if reduce_dim:
            kl = kl.sum(1)
        out = -1 * kl
        out[t == 0] = rec_ll[t==0]
        return out

    def _predict_z0_from_eps(self, z_t, t, eps):
        return (
                _extract_into_tensor( np.sqrt(1.0 / self.alphas_cumprod), t, z_t.shape) * z_t
                - _extract_into_tensor(np.sqrt(1.0 / self.alphas_cumprod - 1), t, z_t.shape) * eps
        )


class DiffusionDCTPrior(DiffusionPrior):
    def __init__(self,
                 model,
                 T,
                 beta_schedule,
                 dct_scale,
                 t_sample='uniform',
                 parametrization='x',
                 num_bits=5,
                 ):
        super(DiffusionDCTPrior, self).__init__(model, T, beta_schedule, t_sample, parametrization, num_bits)
        """
        dct_scale - tensor of the size [ch, h, w] which was used to scale DCT tp the range [-1, 1]. Will be used to scale gaussina noise accordingly
        """
        self.dct_scale = dct_scale
        self.init_schedules()

    def get_beta_schedule(self, name, s=0.008):
        if name == "cosine":
            betas = []
            max_beta = 0.999
            min_beta = 0.001
            fn = lambda t: np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            for i in range(self.T):
                t1 = i / self.T
                t2 = (i + 1) / self.T
                betas.append(np.clip(1 - fn(t2) / fn(t1), min_beta, max_beta))
            return np.stack(betas)
        else:
            raise NotImplementedError(f"unknown beta schedule: {name}")

    def init_schedules(self):
        self.BETAS_mu = self.get_beta_schedule(self.beta_schedule)
        self.alphas_cumprod_mu = np.cumprod(1.0 - self.BETAS_mu, axis=0)

        # beta_scale = self.dct_scale / self.dct_scale.min() #(self.dct_scale ** 0.5)
        # beta_scale = (self.dct_scale ** 0.5)
        # self.BETAS_sigma = self.BETAS_mu[:, None, None, None] / beta_scale
        # self.BETAS_sigma[-1] = self.BETAS_sigma[-1] * beta_scale #self.BETAS_mu[-1] #torch.ones_like(self.BETAS_sigma[-1]) * 0.9
        s = - 0.2 * torch.ones_like(self.dct_scale)
        self.BETAS_sigma = self.get_beta_schedule(self.beta_schedule, s)
        self.BETAS_sigma[0] = self.BETAS_mu[0]
        self.BETAS_sigma = torch.from_numpy(self.BETAS_sigma).float()
        self.alphas_cumprod_sigma = np.cumprod(1.0 - self.BETAS_sigma, axis=0)

        alphas_cumprod_sigma_prev = torch.cat(
            [torch.ones((1,) + self.dct_scale.shape), self.alphas_cumprod_sigma[:-1]])
        alpha_mu = torch.from_numpy(1.0 - self.BETAS_mu)[:, None, None, None]

        denominator = alpha_mu * (1.0 - alphas_cumprod_sigma_prev) + self.BETAS_sigma
        self.posterior_variance = self.BETAS_sigma * (
                    1.0 - alphas_cumprod_sigma_prev) / denominator

        alphas_cumprod_mu_prev = torch.cat(
            [torch.ones(1),
             torch.from_numpy(self.alphas_cumprod_mu[:-1])])[:, None, None, None]
        self.posterior_mean_coef1 = torch.sqrt(
            alphas_cumprod_mu_prev) * self.BETAS_sigma / denominator

        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_sigma_prev) * alpha_mu.sqrt() / denominator

    # def sample(self, N, t=None):
    #     '''
    #     t stand for temperature. Is not used.
    #     '''
    #     shape = [N, self.model.in_channels,  self.model.image_size, self.model.image_size]
    #     std_coef =  torch.sqrt(1.0 - self.alphas_cumprod_sigma)[-1].float().to(self.device)
    #     img = torch.randn(*shape, device=self.device) * std_coef
    #     indices = list(range(self.T))[::-1]
    #     for i in indices:
    #         t = torch.tensor([i] * shape[0], device=self.device)
    #         with torch.no_grad():
    #             img = self.p_sample(img, t)
    #     return img

    def q_sample(self, z_0, t):
        noise = torch.randn_like(z_0)
        mu_coef = _extract_into_tensor(np.sqrt(self.alphas_cumprod_mu), t, z_0.shape)
        std_coef = torch.sqrt(1.0 - self.alphas_cumprod_sigma)[t].float().to(z_0.device)
        return mu_coef * z_0 + std_coef * noise

    def get_q_posterior(self, z_0, z_t, t):
        q_mean = self.posterior_mean_coef1[t].to(z_0.device) * z_0 + \
                 self.posterior_mean_coef2[t].to(z_0.device) * z_t
        q_log_variance = torch.log(self.posterior_variance[t]).float().to(z_0.device)
        return Normal(q_mean.float(), q_log_variance)

    def get_p(self, z_t, t):
        if self.parametrization == 'x':
            p_mean, p_logvar_coef = torch.chunk(self.model(z_t, t), 2, dim=1)
            p_dist = Normal(p_mean, p_logvar_coef)
        elif self.parametrization == 'x_var':
            p_mean, p_logvar_coef = torch.chunk(self.model(z_t, t), 2, dim=1)
            min_log =  torch.log(self.posterior_variance[t]).float().to(z_t.device)
            p_dist = Normal(p_mean, min_log)
        elif self.parametrization == 'eps':
            eps = self.model(z_t, t)
            z0_pred = self._predict_z0_from_eps(z_t, t, eps)
            p_dist = self.get_q_posterior(z0_pred, z_t, t)
        return p_dist

    def _predict_z0_from_eps(self, z_t, t, eps):
        eps_coef = torch.sqrt(1.0 - self.alphas_cumprod_sigma)[t].float().to(z_t.device)
        coef = _extract_into_tensor(1.0 /  np.sqrt(self.alphas_cumprod_mu), t, z_t.shape)
        return coef * (z_t - eps_coef * eps)



def discretized_gaussian_log_likelihood(x, *, means, log_scales, n_bits = 5):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities
    """
    bins = 2 ** n_bits - 1
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / bins )
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / bins)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-10))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-10))
    log_cdf_delta = torch.log((cdf_plus - cdf_min).clamp(min=1e-10))
    # print(x.min(), x.max())
    # print(torch.sum(x <= -1. + 1./255.) / torch.sum(x >= -10.), log_cdf_plus.mean().item())
    # print(torch.sum(x >= 1. - 1. / 255.) / torch.sum(x >= -10.), log_one_minus_cdf_min.mean().item())
    # print(log_cdf_delta.mean().item())
    # breakpoint()
    # print(inv_stdv.mean().item(), centered_x.abs().mean().item())
    log_probs = torch.where(
        x <= -1. + 1./bins,
        log_cdf_plus,
        torch.where(x >= 1. - 1./bins,
                    log_one_minus_cdf_min,
                    log_cdf_delta),
    )
    assert log_probs.shape == x.shape
    return log_probs


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

