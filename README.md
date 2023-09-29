# Discouraging Posterior Collapse in Hierarchical Variational Autoencoders Using Context

This repository contains the code for the experiments from the paper:

[Discouraging Posterior Collapse in Hierarchical Variational Autoencoders Using Context](https://arxiv.org/abs/2302.09976)

**[Anna Kuzina](https://akuzina.github.io/), [Jakub M. Tomczak](https://jmtomczak.github.io/)**

---
## Abstract
<table>
<tr>
<td style="width:100%">
<i>
Hierarchical Variational Autoencoders (VAEs) are among the most popular likelihood-based generative models. 
There is a consensus that the top-down hierarchical VAEs allow effective learning of deep latent structures and avoid problems like posterior collapse. 
Here, we show that this is not necessarily the case, and the problem of collapsing posteriors remains. 
To discourage this issue, we propose a deep hierarchical VAE with a context on top. 
Specifically, we use a Discrete Cosine Transform to obtain the last latent variable.
In a series of experiments, we observe that the proposed modification allows us to achieve better utilization of the latent space and does not harm the model's generative abilities. 
</i>
</table>

## Experiments 
### Environment Setup
We list the required packages in `conda_requirements.yaml`:
```bash
conda env create -f conda_requirements.yaml
```

### Weights and Biases Setup
We use [w&b](https://wandb.ai) to track experiments, save and load trained models, thus it is required to run the code. 
Variables `entity`, `project` and `api_key` in the `configs/wandb/defaults.yaml` file should be specified before running the experiments.  

### Run experiments
Configuration for all the experiments are provided in `configs/experiments` folder. 

For example, to run the experiment with the MNIST dataset and the DCT-based hierarchical VAE, run the following command:
```bash
python -u run_experiment.py experiment=mnist_dct_ladder
```

We use 4-GPUs to train model on the CIFAR-10 dataset (make sure that `train.ddp: True`):
```bash
mpiexec -n 4 python -u run_experiment.py experiment=cifar_dct_ladder
```

### Pre-trained model

Dataset | Model        | Test NLL/BPD | Link
--- |--------------|--------------| --- 
MNIST | DCT-VAE | 76.62        | [link](https://drive.google.com/file/d/1RZWo7jDjA3pPcfZ--qxtBc7GpsOH0Ea4/view?usp=sharing)
OMNIGLOT | DCT-VAE   | 86.11        | [link](https://drive.google.com/file/d/1RbOsL4nyF3nvaO3-n0YoqwejBYSNmvww/view?usp=sharing)
CIFAR-10 | DCT-VAE   | 3.26         | [link](https://drive.google.com/file/d/1Rbl7X09Gr4zXSuUQvuwHcYEjpopeb9Uq/view?usp=sharing)
<!-- SVHN | DCT-VAE   | 1.97         | [link](https://drive.google.com/file/d/1RbwLj0EO7RwfiPqLiL0NwMAsSGzCWyqd/view?usp=sharing) -->

### Cite
If you found this work useful in your research, please consider citing:

```text
@article{kuzina2023discouraging,
  title={Discouraging Posterior Collapse in Hierarchical Variational Autoencoders Using Context},
  author={Kuzina, Anna and Tomczak, Jakub M},
  journal={},
  year={2023}
}
```
### Acknowledgements
This research was (partially) funded by the Hybrid Intelligence Center, a 10-year programme funded by the Dutch Ministry of Education, Culture and Science through the Netherlands Organisation for Scientific Research, https://hybrid-intelligence-centre.nl.

This work was carried out on the Dutch national infrastructure with the support of SURF Cooperative.
