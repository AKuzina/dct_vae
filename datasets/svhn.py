import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision import datasets
import os
from datasets.mnists import MNIST
from datasets.dct import DCT_dataset

from PIL import Image


class Normalize:
    def __init__(self, dequant=False, num_bits=8):
        self.dequant = dequant
        self.num_bits = num_bits

    def __call__(self, x):
        x = torch.FloatTensor(np.asarray(x, dtype=np.float32)).permute(2, 0, 1)
        # shift_loss = -127.5
        # scale_loss = 1. / 127.5
        # dequantize and scale to [0, 1]
        if self.dequant:
            x = (x + torch.rand_like(x).detach()) / (2 ** self.num_bits)
            x = 2 * x - 1
        else:
            x = (x - 127.5)/127.5
            # x = x / (2 ** self.num_bits - 1) #[0, 255] -> [0, 1]

        # map to [-1, 1]
        # return 2*x - 1
        return x


class svhn(datasets.SVHN):
    def __init__(self, root, split, transform=None, download=False):
        root = os.path.join(root, 'SVHN')
        super(svhn, self).__init__(root, split=split, transform=transform, download=download)
        self.train = False
        if split == 'train':
            self.train = True

    @property
    def processed_folder(self) -> str:
        return self.root


class SVHN(MNIST):
    def __init__(self, batch_size, test_batch_size, model, ctx_size, root, mode, ddp=False, mpi_size=None, rank=None):
        super().__init__(batch_size, test_batch_size, model, ctx_size, root, mode, ddp, mpi_size, rank)
        self.transforms = transforms.Compose([
            Normalize(dequant=False)
        ])
        self.test_transforms = transforms.Compose([
            Normalize(dequant=False)
        ])

    def prepare_data(self):
        svhn(self.root, split='train', download=True)
        svhn(self.root, split='test', download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            svhn_full = svhn(self.root, split='train', transform=self.transforms)
            if 'context' in self.model:
                svhn_full = DCT_dataset(svhn_full, self.ctx_size, mode=self.mode)
            N = 73257
            self.train, self.val = random_split(svhn_full, [N-5000, 5000])

        if stage == 'test' or stage is None:
            self.test = svhn(self.root, split='test', transform=self.test_transforms)
            if 'context' in self.model:
                self.test = DCT_dataset(self.test, self.ctx_size, mode=self.mode)

