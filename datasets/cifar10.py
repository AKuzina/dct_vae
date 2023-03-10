import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision import datasets
import hydra
import os
from datasets.mnists import MNIST
from datasets.dct import DCT_dataset
from datasets.svhn import Normalize


class CIFAR10(MNIST):
    def __init__(self, batch_size, test_batch_size, model, ctx_size, root, mode, ddp=False, mpi_size=None, rank=None):
        super().__init__(batch_size, test_batch_size, model, ctx_size, root, mode, ddp, mpi_size, rank)
        self.transforms = transforms.Compose([
            Normalize(dequant=False),
            transforms.RandomHorizontalFlip(),
        ])
        self.test_transforms = transforms.Compose([
            Normalize(dequant=False)
        ])

    def prepare_data(self):
        datasets.CIFAR10(self.root, train=True, download=True)
        datasets.CIFAR10(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            cifar_full = datasets.CIFAR10(self.root, train=True, transform=self.transforms)
            cifar_full.processed_folder = os.path.join(self.root, cifar_full.base_folder)

            if 'context' in self.model:
                cifar_full = DCT_dataset(cifar_full, self.ctx_size, mode=self.mode)
            N = len(cifar_full)
            self.train, self.val = random_split(cifar_full, [N-5000, 5000])

        if stage == 'test' or stage is None:
            self.test = datasets.CIFAR10(self.root, train=False, transform=self.test_transforms)
            self.test.processed_folder = os.path.join(self.root, self.test.base_folder)
            if 'context' in self.model:
                self.test = DCT_dataset(self.test, self.ctx_size, mode=self.mode)

