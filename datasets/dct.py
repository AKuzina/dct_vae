import math
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


def RGB_to_YCBCR(x):
    # [-1, 1] to [0, 1]
    x = (x+1)/2
    # PIL image
    x_pil = transforms.ToPILImage(mode='RGB')(x)
    # convert to YCbCr
    x_y = np.array(x_pil.convert('YCbCr'))
    # map to [0, 1]
    x_y = torch.FloatTensor(x_y).permute(2, 0, 1) / 255.
    return x_y * 2 - 1


def YCBCR_to_RGB(x):
    MB, _, h, w = x.shape
    tr_mat = torch.FloatTensor([[1, 0, 1.5748],
                                [1, -0.1873, -0.4681],
                                [1, 1.8556, 0]]).unsqueeze(0)
    tr_mat = tr_mat.to(x.device)
    x_rgb = torch.clamp(torch.matmul(tr_mat.repeat(MB, 1, 1),
                                     x.reshape(MB, 3, -1)).reshape(MB, 3, h, w), -1, 1)
    return x_rgb

# def YCBCR_to_RGB(x):
#     # [-1, 1] to [0, 1]
#     x = (x+1)/2
#     # PIL image
#     x_pil = transforms.ToPILImage(mode='YCbCr')(x)
#     # convert to YCbCr
#     x_y = np.array(x_pil.convert('RGB'))
#     # map to [0, 1]
#     x_y = torch.FloatTensor(x_y).permute(2, 0, 1) / 255.
#     return x_y * 2 - 1


class DCT(nn.Module):
    def __init__(self, im_width=8, im_height=8, mode='RGB', type='old'):
        super(DCT, self).__init__()
        self.mode = mode
        self.type = type
        assert type in ['old', 'ortho']

        # forward DCT
        A_w = math.pi * (torch.arange(0, im_width) + 0.5) / im_width
        A_h = math.pi * (torch.arange(0, im_height) + 0.5) / im_height
        ints_w = torch.arange(0, im_width)
        ints_h = torch.arange(0, im_height)
        self.basis_function_w = nn.Parameter(
            torch.cos(torch.einsum('ij,jk->ik', ints_w.unsqueeze(1), A_w.unsqueeze(0))),
            requires_grad=False
        )
        self.basis_function_h = nn.Parameter(
            torch.cos(torch.einsum('ij,jk->ik', ints_h.unsqueeze(1), A_h.unsqueeze(0))),
            requires_grad=False
        )
        # inverse DCT
        B_w = (math.pi * (torch.arange(0, im_width) + 0.5) / im_width)
        B_h = (math.pi * (torch.arange(0, im_height) + 0.5) / im_height)

        indx_w = torch.arange(1, im_width)
        indx_h = torch.arange(1, im_height)

        self.reverse_function_w = nn.Parameter(
            torch.cos(torch.einsum('ij,jk->ik', B_w.unsqueeze(1), indx_w.unsqueeze(0))),
            requires_grad=False
        )
        self.reverse_function_h = nn.Parameter(
            torch.cos(torch.einsum('ij,jk->ik', B_h.unsqueeze(1), indx_h.unsqueeze(0))),
            requires_grad=False
        )

    def dct2(self, x):
        assert len(x.shape) == 4
        # covert RGB to YCbCr if required
        if self.mode == 'YCbCr':
            ims = []
            for im in x:
                ims.append(RGB_to_YCBCR(im))
            x = torch.stack(ims, 0)

        # map to [0, 1]
        if x.min() < 0:
            x = 0.5 * (x + 1)
        # x - B x C x H x W
        s = x.shape
        # X - B*C*H x W
        f = 2. * torch.einsum('ij,jk->ik', x.reshape(s[0]*s[1]*s[2], s[3]),
                              self.basis_function_w.t())
        if self.type == 'ortho':
            # normalize
            w = self.basis_function_w.shape[1]
            f[:, 0] = f[:, 0] / math.sqrt(4 * w)
            f[:, 1:] = f[:, 1:] / math.sqrt(2 * w)
        # B*C*H x W -> B*C*W x H
        f = f.reshape(*s).permute(0, 1, 3, 2).reshape(s[0]*s[1]*s[3], s[2])
        F = 2. * torch.einsum('ij,jk->ik', f, self.basis_function_h.t())
        if self.type == 'ortho':
            # normalize
            h = self.basis_function_h.shape[1]
            F[:, 0] = F[:, 0] / math.sqrt(4 * h)
            F[:, 1:] = F[:, 1:] / math.sqrt(2 * h)
        F = F.reshape(s[0], s[1], s[3], s[2]).permute(0, 1, 3, 2)
        return F

    def idct2(self, x):
        assert len(x.shape) == 4
        # x - B x C x H x W
        s = x.shape
        # X - B*C*H x W
        x = x.reshape(s[0]*s[1]*s[2], s[3])
        if self.type == 'old':
            f = (x[:, [0]] + 2. * torch.einsum('ij,jk->ik', x[:, 1:], self.reverse_function_w.t())) / (2.*s[3])
        elif self.type == 'ortho':
            f = (x[:, [0]] + math.sqrt(2.) * torch.einsum('ij,jk->ik', x[:, 1:],
                                                      self.reverse_function_w.t())) / (math.sqrt(s[3]))

        # B*C*H x W -> B*C*W x H
        f = f.reshape(*s).permute(0, 1, 3, 2).reshape(s[0]*s[1]*s[3], s[2])
        if self.type == 'old':
            F = (f[:, [0]] + 2. * torch.einsum('ij,jk->ik', f[:, 1:], self.reverse_function_h.t())) / (2.*s[2])
        elif self.type == 'ortho':
            F = (f[:, [0]] + math.sqrt(2.) * torch.einsum('ij,jk->ik', f[:, 1:],
                                                          self.reverse_function_h.t())) / (math.sqrt(s[2]))
        F = F.reshape(s[0], s[1], s[3], s[2]).permute(0, 1, 3, 2)
        return F


class DCT_dataset(Dataset):
    def __init__(self, base_dataset, ctx_size, mode='RGB'):
        self.base_dataset = base_dataset
        self.ctx_size = ctx_size
        _, h, w = base_dataset[0][0].shape
        self.dct = DCT(h, w, mode=mode)
        self.x_dim = h
        assert mode in ['RGB', 'YCbCr', 'BW'], f'Mode should be BW, RGB or YCbCr, got {mode} instead'
        self.mode = mode
        if hasattr(self.base_dataset, 'train'):
            self.file_name = f'training_dct_{mode}.pt' if self.base_dataset.train else f'test_dct_{mode}.pt'
        elif hasattr(self.base_dataset, 'split'):
            if self.base_dataset.split == 'train':
                self.file_name = f'training_dct_{mode}.pt'
            elif self.base_dataset.split == 'valid':
                self.file_name = f'valid_dct_{mode}.pt'
            else:
                self.file_name = f'test_dct_{mode}.pt'

        path = os.path.join(self.base_dataset.processed_folder, self.file_name)
        if os.path.exists(path):
            self.dct_data = torch.load(path)
        else:
            self.dct_data = self.calculate_dct()
        self.preprocess_dct()

    def preprocess_dct(self):
        # crop the context
        self.dct_data = self.dct_data[:, :, :self.ctx_size, :self.ctx_size]
        # compute stats for normalization
        self.mean = self.dct_data.mean(0)
        self.std = self.dct_data.std(0)
        self.scale = torch.floor(self.dct_data).abs().max(0)[0]

    def calculate_dct(self):
        dloader = DataLoader(self.base_dataset, batch_size=1, drop_last=False, shuffle=False)
        res = []
        for x, _ in dloader:
            res.append(self.dct.dct2(x))
        all_dcts = torch.cat(res)
        if not os.path.exists(self.base_dataset.processed_folder):
            os.makedirs(self.base_dataset.processed_folder)
        torch.save(all_dcts,
                   os.path.join(self.base_dataset.processed_folder, self.file_name))
        return all_dcts

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, item):
        batch = self.base_dataset[item]
        x = batch[0]
        x_dct = self.dct.dct2(x.unsqueeze(0))[0, :, :self.ctx_size, :self.ctx_size]
        batch += (x_dct, )
        return batch

