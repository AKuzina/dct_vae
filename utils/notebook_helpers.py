import os
import wandb
import torch
import numpy as np

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from utils.wandb import api, get_checkpoint

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif" : "Times New Roman",
    "font.size": 34,
    "lines.linewidth":3
}

save_fig_opt = {
    'dpi': 800,
    'transparent':True,
    'bbox_inches':'tight',
    'pad_inches': 0
}

USER = 'anna_jakub'
PROJECT = 'context_vae'


def get_vals(i, filename):
    run_pth = os.path.join(USER, PROJECT, i)
    run = api.run(run_pth)

    file = wandb.restore(f'{filename}.pt', run_path=run_pth, replace=True,
                         root='_loaded/')
    return torch.load(file.name, map_location='cpu')


def line_with_std(arr, ax, label='', plot_std=True):
    m = arr.mean(0)
    s = arr.std(0)
    n_sq = np.sqrt(arr.shape[0])
    ax.plot(m, label=label)
    if plot_std:
        ax.fill_between(range(len(m)), y1=m - 2 * s/ n_sq, y2=m + 2 * s / n_sq, alpha=0.2)
    ax.grid();


def get_im(tensor):
    return tensor.permute(1, 2, 0).detach().numpy()


def get_psnr(mse, max_val=1.):
    if not isinstance(mse, torch.Tensor):
        mse = torch.from_numpy(mse)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def get_jpeg_compressed(idx, quality):
    image = transforms.ToTensor()(Image.open(f'datasets/kodak/kodim{idx}.png'))
    file = 'compressed.jpg'

    # map to [0, 255.]
    image = 255. * image
    pil_im = image.squeeze().permute(1, 2, 0).detach().numpy().astype(np.uint8)
    pil_im = to_pil_image(pil_im)
    pil_im.save(file, subsampling=0, quality=int(quality))

    image_compressed = torch.FloatTensor(np.array(Image.open(file))).permute(2, 0, 1)
    image_compressed = image_compressed / 255.
    return image / 255., image_compressed

def get_full_bpp(idx):
    idx += 1
    if idx < 10:
        idx = '0' + str(idx)
    file = f'datasets/kodak/kodim{idx}.png'
    image = transforms.ToTensor()(Image.open(f'datasets/kodak/kodim{idx}.png'))
    bpp = (os.path.getsize(file) * 8.) / (
            image.shape[1] * image.shape[2])
    return bpp