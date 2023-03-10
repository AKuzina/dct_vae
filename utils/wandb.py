import wandb
import os
import torch
import omegaconf
from hydra.utils import instantiate


api = wandb.Api()


def get_checkpoint(wandb_args, idx, device='cpu'):
    # download the checkpoint from wandb to the local machine.
    file = wandb.restore('last_chpt.pth',
                         run_path=os.path.join(wandb_args.entity,
                                               wandb_args.project,
                                               idx),
                         replace=True)
    # load the checkpoint
    chpt = torch.load(file.name, map_location=device)
    return chpt


def load_model(idx, wandb_args):
    pth = os.path.join(wandb_args.entity, wandb_args.project, idx)
    run = api.run(pth)
    config = omegaconf.OmegaConf.create(run.config)

    # LOAD THE MODEL
    vae = instantiate(config.model)
    file = wandb.restore('last_chpt.pth', run_path=pth, replace=True)
    chpt = torch.load(file.name, map_location='cpu')

    if chpt['ema_model_state_dict'] is not None:
        vae.load_state_dict(chpt['ema_model_state_dict'])
    else:
        vae.load_state_dict(chpt['model_state_dict'])
    return vae, config


def load_data_module(idx, wandb_args, test_batch_size=None):
    pth = os.path.join(wandb_args.entity, wandb_args.project, idx)
    run = api.run(pth)
    config = omegaconf.OmegaConf.create(run.config)
    dset_params = {
            'root': 'datasets/'
        }
    if 'context' in config.model.name:
        dset_params['ctx_size'] = config.model.ctx_size
    if test_batch_size is not None:
        config.dataset.data_module.test_batch_size = test_batch_size
    data_module = instantiate(config.dataset.data_module, **dset_params)
    return data_module
