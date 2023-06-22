import torch
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger
import logging
from ray import tune
import ray
from copy import deepcopy
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Union

from src.models.utils import subset_by_indices, NormalizedRBF

logger = logging.getLogger(__name__)


def fit_eval_kfold(args: dict, orig_hparams: DictConfig, model_cls, train_data_dict: dict, **kwargs):
    """
    Globally defined method, used for hyperparameter tuning with ray
    :param args: Hyperparameter configuration
    :param orig_hparams: DictConfig of original hyperparameters
    :param model_cls: class of model
    :param kwargs: Other args
    """
    new_params = deepcopy(orig_hparams)
    model_cls.set_hparams(new_params.model, args)

    kf = KFold(n_splits=5, random_state=orig_hparams.exp.seed, shuffle=True)
    val_metrics = []
    for train_index, val_index in kf.split(train_data_dict['cov_f']):
        ttrain_data_dict, val_data_dict = \
            subset_by_indices(train_data_dict, train_index), subset_by_indices(train_data_dict, val_index)

        model = model_cls(new_params, **kwargs)
        model.fit(train_data_dict=ttrain_data_dict, log=False)
        tune_criterion = getattr(model, model.tune_criterion)
        val_metric = tune_criterion(val_data_dict['treat_f'], val_data_dict['out_f'], val_data_dict['cov_f']).mean()
        val_metrics.append(val_metric)

    tune.report(val_metric=np.mean(val_metrics))


class InterventionalDensityEstimator(torch.nn.Module):
    """
    Abstract class for an interventional density esimator
    """

    tune_criterion = None

    def __init__(self, args: DictConfig = None, **kwargs):
        super(InterventionalDensityEstimator, self).__init__()

        # Dataset params
        self.dim_out, self.dim_cov, self.dim_treat = args.model.dim_out, args.model.dim_cov, args.model.dim_treat
        self.treat_options = np.array(args.model.treat_options, dtype=float)
        assert self.dim_treat == 1

        # Model hyparams
        self.hparams = args

        # MlFlow Logger
        if args.exp.logging:
            experiment_name = f'{args.model.name}/{args.dataset.name}'
            self.mlflow_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)

    def prepare_train_data(self, train_data_dict: dict):
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        raise NotImplementedError()

    def prepare_tensors(self, cov=None, treat=None, out=None, kind='torch') -> Tuple[dict]:
        """
        Conversion of tensors
        @param cov: Tensor with covariates
        @param treat: Tensor with treatments
        @param out: Tensor with outcomes
        @param kind: torch / numpy
        @return: cov, treat, out
        """
        if kind == 'torch':
            cov = torch.tensor(cov).reshape(-1, self.dim_cov).float() if cov is not None else None
            treat = torch.tensor(treat).reshape(-1, self.dim_treat).float() if treat is not None else None
            out = torch.tensor(out).reshape(-1, self.dim_out).float() if out is not None else None
        elif kind == 'numpy':
            cov = cov.reshape(-1, self.dim_cov) if cov is not None else None
            treat = treat.reshape(-1).astype(float) if treat is not None else None
            out = out.reshape(-1, self.dim_out) if out is not None else None
        else:
            raise NotImplementedError()
        return cov, treat, out

    def fit(self, train_data_dict: dict, log: bool) -> None:
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        raise NotImplementedError()

    def inter_log_prob(self, treat_pot, out_pot) -> Union[np.array, torch.Tensor]:
        """
        Interventional log-probability
        @param treat_pot: Tensor of potential treatments
        @param out_pot: Tensor of potential outcome values
        @return: Tensor with log-probabilities
        """
        raise NotImplementedError()

    def inter_sample(self, treat_pot, n_samples) -> Union[np.array, torch.Tensor]:
        """
        Sampling from the estimated interventional density
        @param treat_pot: Tensor of potential treatments
        @param n_samples: number of samples
        @return: Tensor with sampled data
        """
        raise NotImplementedError()

    def inter_mean(self, treat_option, **kwargs) -> Union[np.array, torch.Tensor]:
        """
        Mean of potential outcomes
        @param treat_option: Treatment 0 / 1
        @return: mean
        """
        raise NotImplementedError()

    def save_train_data_to_buffer(self, cov_f, treat_f, out_f) -> None:
        """
        Save train data for non-parametric inference of two-stage training
        @param cov_f: Tensor with factual covariates
        @param treat_f: Tensor with factual treatments
        @param out_f: Tensor with factual outcomes
        """
        self.cov_f = cov_f
        self.treat_f = treat_f
        self.out_f = out_f

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict) -> None:
        """
        Set hyperparameters during tuning
        @param model_args: Default hparameters
        @param new_model_args: Hparameters for tuning
        """
        raise NotImplementedError()

    def set_norm_consts(self, add_bound=2.5) -> None:
        """
        Calculating normalization constants for improperly normalized density estimators
        @param add_bound: Additional bounds for normalization of truncated series estimator
        """

        self.norm_const = [1.0, 1.0]
        logger.info('Calculating normalization constants')

        for i, treat_option in enumerate(self.treat_options):
            if self.dim_out == 1:
                norm_bins_effect = self.norm_bins
                out_pot = np.linspace(self.out_f.min() - add_bound, self.out_f.max() + add_bound, norm_bins_effect)
                dx = (self.out_f.max() - self.out_f.min() + 2 * add_bound) / norm_bins_effect
            elif self.dim_out == 2:
                norm_bins_sqrt = int(np.sqrt(self.norm_bins)) + 1
                norm_bins_effect = norm_bins_sqrt ** 2
                out_pot_0 = np.linspace(self.out_f[:, 0].min() - add_bound, self.out_f[:, 0].max() + add_bound, norm_bins_sqrt)
                out_pot_1 = np.linspace(self.out_f[:, 1].min() - add_bound, self.out_f[:, 1].max() + add_bound, norm_bins_sqrt)
                out_pot_0, out_pot_1 = np.meshgrid(out_pot_0, out_pot_1)
                out_pot = np.concatenate([out_pot_0.reshape(-1, 1), out_pot_1.reshape(-1, 1)], axis=1)
                dx = (self.out_f[:, 0].max() - self.out_f[:, 0].min() + 2 * add_bound) *\
                     (self.out_f[:, 1].max() - self.out_f[:, 1].min() + 2 * add_bound) / norm_bins_effect
            else:
                raise NotImplementedError()
            treat_pot = treat_option * np.ones((norm_bins_effect,))
            log_prob = self.inter_log_prob(treat_pot, out_pot)
            if isinstance(log_prob, torch.Tensor):
                log_prob = log_prob.detach().numpy()
            prob = np.exp(log_prob)
            self.norm_const[i] = prob.sum() * dx

    def set_sd_y_median_heuristic(self) -> None:
        """
        Calculate median heuristics (for KDE and DKME)
        """
        self.sd_y = {}
        for treat_option in self.treat_options:
            distances = np.tril(squareform(pdist(self.out_f[self.treat_f.reshape(-1) == treat_option].reshape(-1, self.dim_out),
                                                 'sqeuclidean')), -1)
            self.sd_y[treat_option] = np.median(distances[distances > 0.0])
            self.normalized_rbf_y[treat_option] = NormalizedRBF(self.sd_y[treat_option])
            logger.info(f'New sd_y[{treat_option}]: {self.sd_y[treat_option]}')

    def finetune(self, train_data_dict: dict, resources_per_trial: dict):
        """
        Hyperparameter tuning with ray[tune]
        @param train_data_dict: Training data dictionary
        @param resources_per_trial: CPU / GPU resources dictionary
        @return: self
        """

        logger.info(f"Running hyperparameters selection with {self.hparams.model['tune_range']} trials")
        logger.info(f'Using {self.tune_criterion} for hyperparameters selection')
        ray.init(num_gpus=0, num_cpus=2)

        hparams_grid = {k: getattr(tune, self.hparams.model['tune_type'])(list(v))
                        for k, v in self.hparams.model['hparams_grid'].items()}
        analysis = tune.run(tune.with_parameters(fit_eval_kfold,
                                                 model_cls=self.__class__,
                                                 train_data_dict=deepcopy(train_data_dict),
                                                 orig_hparams=self.hparams),
                            resources_per_trial=resources_per_trial,
                            raise_on_failed_trial=False,
                            metric="val_metric",
                            mode="max",
                            config=hparams_grid,
                            num_samples=self.hparams.model['tune_range'],
                            name=f"{self.__class__.__name__}",
                            max_failures=1,
                            )
        ray.shutdown()

        logger.info(f"Best hyperparameters found: {analysis.best_config}.")
        logger.info("Resetting current hyperparameters to best values.")
        self.set_hparams(self.hparams.model, analysis.best_config)

        self.__init__(self.hparams)
        return self
