# INFs
Normalizing Flows for Interventional Density Estimation

<img width="1335" alt="image" src="https://github.com/Valentyn1997/INFs/assets/23198776/a3557641-9ac4-48de-b596-4c6679e56fc2">

The project is built with following Python libraries:
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

To access MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to local browser http://localhost:5000.


## Experiments

Main training script is universal for different methods and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `configs/` folder.

Generic script with logging and fixed random seed is following:
```console
PYTHONPATH=.  python3 runnables/train.py +dataset=<dataset> +model=<backbone> exp.seed=10
```

### Models (baselines)
One needs to choose a model and then fill the specific hyperparameters (they are left blank in the configs):
- Interventional Normalizing Flows (this paper): `+model=interflow`
  - main: `+model=aiptw_teacher_student`
  - w/o stud flow (= [Conditional Normalizing Flow](https://arxiv.org/pdf/1802.04908.pdf)): `+model=marginalized_teacher`
  - w/o bias corr: `+model=teacher_student`
- [Conditional Normalizing Flows + Truncated Series Estimator](https://academic.oup.com/biomet/advance-article-abstract/doi/10.1093/biomet/asad017/7068801?redirectedFrom=fulltext) (CNF + TS): `+model=aiptw_truncated_series`
- [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) (MDNs): `+model=mdn`
- [Kernel Density Estimation](https://arxiv.org/pdf/1806.02935.pdf) (KDE): `+model=kde`
- [Distributional Kernel Mean Embeddings](https://arxiv.org/pdf/1805.08845.pdf) (DKME): `+model=kme`
- TARNet* (extended distributional [TARNet](https://arxiv.org/abs/1606.03976)): `+model=gauss_tarnet`

Models already have best hyperparameters saved (for each model and dataset), one can access them via: `+model/<dataset>_hparams=<model>` or `+model/<dataset>_hparams/<model>=<dataset_param>`. Hyperparameters for three variants of INFs are the same.

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
Example of running 10-fold run with INFs (main) on Synthetic data with b = [2.0, 3.0, 4.0]:
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=polynomial_normal +model=aiptw_teacher_student +model/polynomial_normal_hparams/interflow='2.0','3.0','4.0' exp.seed=10
```

Example of running 5 runs with random splits with MDNs on the first subset of ACIC 2016 (with tuning, based on the first split):
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=acic_2016 +model=mdn model.tune_hparams=True exp.seed=10 dataset.n_shuffle_splits=5 dataset.dataset_ix=0
```

Example of running 10 runs with random splits with INFs (main) on HC-MNIST:
```console
PYTHONPATH=. python3 runnables/train.py -m +dataset=hcmnist +model=aiptw_teacher_student +model/hcmnist_hparams=interflow model.student_count_bins=10 exp.seed=101 model.num_epochs=15000 model.student_num_epochs=5000 model.teacher_hid_dim_multiplier=30 dataset.n_shuffle_splits=10
```

