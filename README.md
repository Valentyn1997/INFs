# INFs
[![Conference](https://img.shields.io/badge/ICML23-Paper-blue])](https://proceedings.mlr.press/v202/melnychuk23a/melnychuk23a.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2209.06203-b31b1b.svg)](https://arxiv.org/abs/2209.06203)
[![Python application](https://github.com/Valentyn1997/INFs/actions/workflows/python-app.yml/badge.svg)](https://github.com/Valentyn1997/INFs/actions/workflows/python-app.yml)

Normalizing Flows for Interventional Density Estimation

<img width="1335" alt="image" src="https://github.com/Valentyn1997/INFs/assets/23198776/a3557641-9ac4-48de-b596-4c6679e56fc2">

The project is built with the following Python libraries:
1. [Pyro](https://pyro.ai/) - deep learning and probabilistic models (MDNs, NFs)
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking

### Installations
First one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000`

To access the MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to the local browser http://localhost:5000.


## Experiments

The main training script is universal for different methods and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `config/` folder.

Generic script with logging and fixed random seed is the following:
```console
PYTHONPATH=.  python3 runnables/train.py +dataset=<dataset> +model=<model> exp.seed=10
```

### Models (baselines)
One needs to choose a model and then fill in the specific hyperparameters (they are left blank in the configs):
- Interventional Normalizing Flows (this paper):
  - main: `+model=infs_aiptw`
  - w/o stud flow (= [Conditional Normalizing Flow](https://arxiv.org/pdf/1802.04908.pdf)): `+model=infs_plugin`
  - w/o bias corr: `+model=infs_covariate_adjusted`
- [Conditional Normalizing Flows + Truncated Series Estimator](https://academic.oup.com/biomet/advance-article-abstract/doi/10.1093/biomet/asad017/7068801?redirectedFrom=fulltext) (CNF + TS): `+model=cnf_truncated_series_aiptw`
- [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) (MDNs): `+model=mdn_plugin`
- [Kernel Density Estimation](https://arxiv.org/pdf/1806.02935.pdf) (KDE): `+model=kde`
- [Distributional Kernel Mean Embeddings](https://arxiv.org/pdf/1805.08845.pdf) (DKME): `+model=dkme`
- TARNet* (extended distributional [TARNet](https://arxiv.org/abs/1606.03976)): `+model=gauss_tarnet_plugin`

Models already have the best hyperparameters saved (for each model and dataset), one can access them via: `+model/<dataset>_hparams=<model>` or `+model/<dataset>_hparams/<model>=<dataset_param>`. Hyperparameters for three variants of INFs are the same: `+model/<dataset>_hparams=infs`.

To perform a manual hyperparameter tuning use the flags `model.tune_hparams=True`, and then see `model.hparams_grid`. 

### Datasets
Before running semi-synthetic experiments, place datasets to `data/` folder:
- [IHDP dataset](https://github.com/AMLab-Amsterdam/CEVAE/blob/master/datasets/IHDP/csv/ihdp_npci_1.csv): `ihdp_npci_1.csv`
- [ACIC 2016](https://jenniferhill7.wixsite.com/acic-2016/competition): 
```
 ── acic_2016
    ├── synth_outcomes
    |   ├── zymu_<id0>.csv   
    |   ├── ... 
    │   └── zymu_<id14>.csv 
    ├── ids.csv
    └── x.csv 
```
- [ACIC 2018](https://www.synapse.org/#!Synapse:syn11294478/wiki/486304):
```
 ── acic_2018
    ├── scaling
    |   ├── <id0>.csv 
    |   ├── <id0>_cf.csv
    |   ├── ... 
    │   ├── <id23>.csv
    │   └── <id23>_cf.csv 
    ├── ids.csv
    └── x.csv 
```
We also provide ids of random datasets (`ids.csv`), used in experiments for ACIC 2016 and ACIC 2018.

One needs to specify a dataset / dataset generator (and some additional parameters, e.g. set b for `polynomial_normal` with `dataset.cov_shift=3.0` or dataset index for ACIC with `dataset.dataset_ix=0`):
- Synthetic data using the SCM: `+dataset=polynomial_normal`
- [IHDP](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162) dataset: `+dataset=ihdp`
- ACIC 2016 & 2018 datasets: `+dataset=acic_2016` / `+dataset=acic_2018` 
- [HC-MNIST](https://github.com/anndvision/quince/blob/main/quince/library/datasets/hcmnist.py) dataset: `+dataset=hcmnist`

### Examples
Example of running a 10-fold run with INFs (main) on Synthetic data with b = [2.0, 3.0, 4.0]:
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=polynomial_normal +model=infs_aiptw +model/polynomial_normal_hparams/infs='2.0','3.0','4.0' exp.seed=10
```

Example of running 5 runs with random splits with MDNs on the first subset of ACIC 2016 (with tuning, based on the first split):
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=acic_2016 +model=mdn_plugin model.tune_hparams=True exp.seed=10 dataset.n_shuffle_splits=5 dataset.dataset_ix=0
```

Example of running 10 runs with random splits with INFs (main) on HC-MNIST:
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=hcmnist +model=infs_aiptw +model/hcmnist_hparams=infs model.target_count_bins=10 exp.seed=101 model.num_epochs=15000 model.target_num_epochs=5000 model.nuisance_hid_dim_multiplier=30 dataset.n_shuffle_splits=10
```

