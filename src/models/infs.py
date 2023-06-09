import torch
from torch.distributions import TransformedDistribution
import pyro.distributions as dist
import pyro.distributions.transforms as T
from omegaconf import DictConfig
import logging
import numpy as np
from pyro.nn import DenseNN, AutoRegressiveNN, ConditionalAutoRegressiveNN
import contextlib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch_ema import ExponentialMovingAverage
from typing import Tuple

from src.models import InterventionalDensityEstimator

logger = logging.getLogger(__name__)


class PluginINFs(InterventionalDensityEstimator):
    """
    CNF [=INFs w/o target flow]: Semi-parametric plugin interventional density estimator with conditional normalizing flows
    """

    tune_criterion = 'cond_log_prob'

    def __init__(self, args: DictConfig = None, **kwargs):
        super(PluginINFs, self).__init__(args)

        self.cov_scaler = StandardScaler()
        self.device = 'cpu'

        # Model hyparams & Train params
        self.nuisance_count_bins = args.model.nuisance_count_bins
        self.nuisance_bound = args.model.nuisance_bound
        self.nuisance_hid_dim_multiplier = args.model.nuisance_hid_dim_multiplier
        self.noise_std_X, self.noise_std_Y = args.model.noise_std_X, args.model.noise_std_Y
        self.num_epochs = args.model.num_epochs
        self.nuisance_lr = args.model.nuisance_lr
        self.nuisance_batch_size = args.model.batch_size
        self.n_mc = args.model.n_mc  # For the MC mean estimation

        # Model parameters = Conditional NFs (marginalized nuisance)
        self.has_prop_score = False
        self.cond_base_dist = dist.MultivariateNormal(torch.zeros(self.dim_out).float(),
                                                      torch.diag(torch.ones(self.dim_out)).float())

        self.cond_loc = torch.nn.Parameter(torch.zeros((self.dim_out, )).float())
        self.cond_scale = torch.nn.Parameter(torch.ones((self.dim_out, )).float())
        self.cond_affine_transform = T.AffineTransform(self.cond_loc, self.cond_scale)

        self.dim_hid = self.nuisance_hid_dim_multiplier
        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid], param_dims=[self.dim_hid]).float()
        if self.dim_out == 1:
            self.cond_spline_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                          param_dims=[self.nuisance_count_bins,
                                                      self.nuisance_count_bins,
                                                      (self.nuisance_count_bins - 1)]).float()
            self.cond_spline_transform = T.ConditionalSpline(self.cond_spline_nn, self.dim_out,
                                                             order='quadratic',
                                                             count_bins=self.nuisance_count_bins,
                                                             bound=self.nuisance_bound).to(self.device)
        else:
            self.cond_spline_nn = ConditionalAutoRegressiveNN(self.dim_out,
                                                              self.dim_hid + self.dim_treat,
                                                              [self.dim_hid],
                                                              param_dims=[self.nuisance_count_bins,
                                                                          self.nuisance_count_bins,
                                                                          (self.nuisance_count_bins - 1)]).float()
            self.cond_spline_transform = T.ConditionalSplineAutoregressive(self.dim_out,
                                                                           self.cond_spline_nn,
                                                                           order='quadratic',
                                                                           count_bins=self.nuisance_count_bins,
                                                                           bound=self.nuisance_bound).to(self.device)
        self.cond_flow_dist = dist.ConditionalTransformedDistribution(self.cond_base_dist,
                                                                      [self.cond_affine_transform, self.cond_spline_transform])

    def get_train_dataloader(self, cov_f, treat_f, out_f, batch_size) -> DataLoader:
        training_data = TensorDataset(cov_f, treat_f, out_f)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        return train_dataloader

    def get_nuisance_optimizer(self) -> torch.optim.Optimizer:
        """
        Init optimizer for the nuisance flow
        """
        modules = torch.nn.ModuleList([self.repr_nn, self.cond_spline_transform])
        return torch.optim.SGD(list(modules.parameters()) + [self.cond_loc, self.cond_scale], lr=self.nuisance_lr, momentum=0.9)

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict) -> None:
        """
        Set hyperparameters during tuning
        @param model_args: Default hparameters
        @param new_model_args: Hparameters for tuning
        """
        model_args.nuisance_count_bins = new_model_args['nuisance_count_bins']
        model_args.noise_std_X = new_model_args['noise_std_X']
        model_args.noise_std_Y = new_model_args['noise_std_Y']
        model_args.nuisance_lr = new_model_args['nuisance_lr']
        model_args.batch_size = new_model_args['batch_size']

    def prepare_train_data(self, train_data_dict: dict) -> Tuple[torch.Tensor]:
        """
        Data pre-processing
        :param train_data_dict: Dictionary with the training data
        """
        cov_f, treat_f, out_f = self.prepare_tensors(train_data_dict['cov_f'], train_data_dict['treat_f'],
                                                     train_data_dict['out_f'], kind='torch')
        self.hparams.dataset.n_samples_train = cov_f.shape[0]
        return cov_f, treat_f, out_f

    def _get_out_pot_grid(self) -> torch.Tensor:
        """
        @return: Grid with potential outcome values and the bin size
        """
        out_f_min, out_f_max = self.out_f.min(), self.out_f.max()
        out_pot = torch.linspace(out_f_min, out_f_max, self.target_nce_bins).reshape(-1, self.dim_out)
        noised_out_pot = out_pot + self.noise_ce * torch.randn_like(out_pot)
        bin_size = (out_f_max - out_f_min) / self.target_nce_bins
        return noised_out_pot, bin_size

    def fit(self, train_data_dict: dict, log: bool) -> None:
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        # Preparing data
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        cov_f = torch.tensor(self.cov_scaler.fit_transform(cov_f)).float()
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f, self.nuisance_batch_size)

        # Preparing optimizers
        nuisance_optimizer = self.get_nuisance_optimizer()
        self.to(self.device)

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving train train_data_dict
        self.save_train_data_to_buffer(cov_f, treat_f, out_f)

        # Conditional NFs (marginalized nuisance) fitting
        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f = next(iter(train_dataloader))
            cov_f, treat_f, out_f = cov_f.to(self.device), treat_f.to(self.device), out_f.to(self.device)
            nuisance_optimizer.zero_grad()

            # Representation -> Adding noise + concat of factual treatment -> Conditional distribution
            repr_f = self.repr_nn(cov_f)
            noised_out_f = out_f + self.noise_std_Y * torch.randn_like(out_f)
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
            context = torch.cat([noised_repr_f, treat_f], dim=1)
            log_prob = self.cond_flow_dist.condition(context).log_prob(noised_out_f)

            loss = - log_prob.mean()
            loss.backward()

            nuisance_optimizer.step()
            self.cond_flow_dist.clear_cache()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics({'train_cond_neg_log_prob': loss.item()}, step=step)

    def cond_log_prob(self, treat_f, out_f, cov_f) -> torch.Tensor:
        """
        Conditional log-probability
        @param treat_f: Tensor with factual treatments
        @param out_f: Tensor with factual outcomes
        @param cov_f: Tensor with factual covariates
        @return: Tensor with log-probabilities
        """
        # Preparing data
        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, treat_f, out_f, kind='torch')
        cov_f = torch.tensor(self.cov_scaler.transform(cov_f)).float()

        with torch.no_grad():
            repr = self.repr_nn(cov_f) if not self.has_prop_score else self.repr_nn(cov_f)[0]
            context = torch.cat([repr, treat_f], dim=1)
            log_prob = self.cond_flow_dist.condition(context).log_prob(out_f)
        return log_prob

    def inter_log_prob(self, treat_pot, out_pot) -> torch.Tensor:
        """
        Interventional log-probability for large datasets
        @param treat_pot: Tensor of potential treatments
        @param out_pot: Tensor of potential outcome values
        @return: Tensor with log-probabilities
        """
        if out_pot.shape[0] > 25000 or self.cov_f.shape[0] > 25000:
            log_prob_pot = torch.zeros((out_pot.shape[0], 1))
            batch_size = 100
            for i in range(out_pot.shape[0] // batch_size + 1):
                log_prob_pot[i * batch_size: (i + 1) * batch_size] = \
                    self.inter_nuisances_log_prob(treat_pot[i * batch_size: (i + 1) * batch_size],
                                                  out_pot[i * batch_size: (i + 1) * batch_size])
            return log_prob_pot
        else:
            return self.inter_nuisances_log_prob(treat_pot, out_pot)

    def inter_nuisances_log_prob(self, treat_pot, out_pot, marginalize=True, with_grad=False, cov_f_batch=None) -> torch.Tensor:
        """
        Interventional log-probability
        @param cov_f_batch: Sample average over a certain batch of covariates
        @param with_grad: Enable gradient computation
        @param marginalize: Sample average over covariates
        @param treat_pot: Tensor of potential treatments
        @param out_pot: Tensor of potential outcome values
        @return: Tensor with log-probabilities
        """
        assert treat_pot.shape[0] == out_pot.shape[0]
        n = out_pot.shape[0]
        _, treat_pot, out_pot = self.prepare_tensors(None, treat_pot, out_pot, kind='torch')

        # Exploding train_data_dict for inference
        cov_f = self.cov_f.repeat(n, 1, 1) if cov_f_batch is None else cov_f_batch.repeat(n, 1, 1)
        treat_pot = treat_pot.unsqueeze(1).repeat(1, cov_f.shape[1], 1)
        out_pot = out_pot.unsqueeze(1).repeat(1, cov_f.shape[1], 1)

        ctx_mgr = torch.no_grad() if not with_grad else contextlib.suppress()
        with ctx_mgr:
            # Propensity is not used
            repr_f = self.repr_nn(cov_f) if not self.has_prop_score else self.repr_nn(cov_f)[0]
            context = torch.cat([repr_f, treat_pot], dim=2)
            log_prob_pot = self.cond_flow_dist.condition(context).log_prob(out_pot)
            if marginalize:
                log_prob_pot = log_prob_pot.exp().mean(1).log()

        return log_prob_pot

    def inter_sample(self, treat_pot, n_samples) -> torch.Tensor:
        """
        Sampling from the estimated interventional density
        @param treat_pot: Tensor of potential treatments
        @param n_samples: number of samples
        @return: Tensor with sampled data
        """
        return self.inter_nuisances_sample(treat_pot, n_samples)

    def inter_nuisances_sample(self, treat_pot, n_samples, cov_f_batch=None, marginalize=True) -> torch.Tensor:
        """
        Sampling from the estimated interventional density
        @param marginalize: Sample average over covariates
        @param cov_f_batch: Sample average over a certain batch of covariates
        @param treat_pot: Tensor of potential treatments
        @param n_samples: number of samples
        @return: Tensor with sampled data
        """
        n = treat_pot.shape[0]
        _, treat_pot, _ = self.prepare_tensors(None, treat_pot, None, kind='torch')
        cov_f = self.cov_f.repeat(n, 1, 1) if cov_f_batch is None else cov_f_batch.repeat(n, 1, 1)
        treat_pot = treat_pot.unsqueeze(1).repeat(1, cov_f.shape[1], 1)

        with torch.no_grad():
            repr_f = self.repr_nn(cov_f) if not self.has_prop_score else self.repr_nn(cov_f)[0]
            context = torch.cat([repr_f, treat_pot], dim=2)
            sampled_out_cond = self.cond_flow_dist.condition(context).sample((n_samples, n, cov_f.shape[1],))
            if marginalize:
                sampled_out_pot = torch.gather(sampled_out_cond, 2,
                                               torch.randint(cov_f.shape[1], (n_samples, n, 1, self.dim_out))).squeeze(2)
                return sampled_out_pot
            else:
                return sampled_out_cond

    def inter_mean(self, treat_option, **kwargs) -> np.array:
        """
        Mean of potential outcomes
        @param treat_option: Treatment 0 / 1
        @return: mean
        """
        treat_pot = treat_option * torch.ones((self.n_mc, ))
        return self.inter_sample(treat_pot, 1).mean((0, 1)).numpy()


class CovariateAdjustedINFs(PluginINFs):
    """
    INFs w/o bias corr: Fully-parametric covariate-adjusted interventional density estimator with (i) conditional normalizing
    flows and (ii) normalizing flows
    """

    tune_criterion = "cond_log_prob"

    def __init__(self, args: DictConfig = None, **kwargs):
        super(CovariateAdjustedINFs, self).__init__(args)

        # Model hyparams & Train params
        self.target_count_bins = args.model.target_count_bins if args.model.target_count_bins is not None \
            else args.model.nuisance_count_bins
        self.target_mode = args.model.target_mode
        self.target_lr = args.model.target_lr
        self.target_nce_bins = args.model.target_nce_bins
        self.target_quadrature = args.model.target_quadrature
        self.noise_ce = args.model.noise_ce
        self.ema_target = None
        self.target_batch_size = 64
        self.target_ema = args.model.target_ema

        self.target_num_epochs = args.model.target_num_epochs

        # Model parameters = NF (target)
        self.base_dist = [dist.MultivariateNormal(torch.zeros(self.dim_out).float(),
                                                  torch.diag(torch.ones(self.dim_out).float())) for _ in self.treat_options]
        self.loc = [torch.nn.Parameter(torch.zeros(self.dim_out).float()) for _ in self.treat_options]
        self.scale = [torch.nn.Parameter(torch.ones(self.dim_out).float()) for _ in self.treat_options]
        self.affine_transform = [T.AffineTransform(loc, scale) for (loc, scale) in zip(self.loc, self.scale)]
        if self.dim_out == 1:
            self.spline_transform = [T.spline(self.dim_out,
                                              order='quadratic',
                                              count_bins=self.target_count_bins,
                                              bound=args.model.target_bound).float() for _ in self.treat_options]
        else:
            self.auto_regressive_nn = [AutoRegressiveNN(self.dim_out, [self.dim_hid],
                                                        param_dims=[self.target_count_bins,
                                                                    self.target_count_bins,
                                                                    (self.target_count_bins - 1)]).float()
                                       for _ in self.treat_options]

            self.spline_transform = [T.SplineAutoregressive(self.dim_out, arn,
                                                            order='quadratic',
                                                            count_bins=self.target_count_bins,
                                                            bound=args.model.target_bound) for arn in self.auto_regressive_nn]
        self.flow_dist = [TransformedDistribution(base_dist, [affine_transform, spline_transform])
                          for (base_dist, affine_transform, spline_transform) in
                          zip(self.base_dist, self.affine_transform, self.spline_transform)]

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict) -> None:
        """
        Set hyperparameters during tuning
        @param model_args: Default hparameters
        @param new_model_args: Hparameters for tuning
        """
        model_args.target_count_bins = new_model_args['target_count_bins']

    def get_target_optimizers(self) -> torch.optim.Optimizer:
        """
        Init optimizer for the target flow
        """
        if self.dim_out == 1:
            modules = [torch.nn.ModuleList([spline_transform]) for spline_transform in self.spline_transform]
        else:
            modules = [torch.nn.ModuleList([arn]) for arn in self.auto_regressive_nn]
        parameters = [list(module.parameters()) + [loc, scale] for (module, loc, scale) in zip(modules, self.loc, self.scale)]
        optimizers = [torch.optim.Adam(par, lr=self.target_lr) for par in parameters]
        ema_target = ExponentialMovingAverage([p for par in parameters for p in par], decay=self.target_ema)
        return optimizers, ema_target

    def cross_entropy(self, flow_dist, treat_option, cov_f) -> torch.Tensor:
        """
        Cross-entropy for the second stage training
        @param flow_dist: Target flow for treatment 0 / 1
        @param treat_option: treatment 0 / 1
        @param cov_f: Batch of factual covariates
        @return: cross-entropy
        """
        treat_pot = treat_option * torch.ones((self.target_nce_bins,))
        if self.dim_out == 1:  # Grid based approximation of cross-entropy
            out_pot = torch.linspace(self.out_f.min(), self.out_f.max(), self.target_nce_bins).reshape(-1, self.dim_out)
            noised_out_pot = out_pot + self.noise_ce * torch.randn_like(out_pot)
            bin_size = (self.out_f.max() - self.out_f.min()) / self.target_nce_bins
            log_prob = flow_dist.log_prob(noised_out_pot).squeeze()
            with torch.no_grad():
                target_log_prob = self.inter_nuisances_log_prob(treat_pot, noised_out_pot, cov_f_batch=cov_f).squeeze()
            if self.target_quadrature == 'rect':
                return - (target_log_prob.exp() * log_prob).sum() * bin_size
            elif self.target_quadrature == 'trap':
                return - 0.5 * (target_log_prob.exp() * log_prob).unfold(0, 2, 1).sum() * bin_size
            else:
                raise NotImplementedError()
        else:  # MC sampling for approximation of cross-entropy
            with torch.no_grad():
                out_pot_sample = self.inter_nuisances_sample(treat_pot, 1, cov_f_batch=cov_f)
            return - flow_dist.log_prob(out_pot_sample).squeeze().mean()

    def fit(self, train_data_dict: dict, log: bool) -> None:
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        # Preparing data
        cov_f_full, treat_f_full, out_f_full = self.prepare_train_data(train_data_dict)
        cov_f_full = torch.tensor(self.cov_scaler.fit_transform(cov_f_full)).float()
        train_dataloader = self.get_train_dataloader(cov_f_full, treat_f_full, out_f_full, self.nuisance_batch_size)

        # Preparing optimizers
        nuisance_optimizer = self.get_nuisance_optimizer()
        target_optimizers, self.ema_target = self.get_target_optimizers()
        self.to(self.device)

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving train train_data_dict
        self.save_train_data_to_buffer(cov_f_full, treat_f_full, out_f_full)

        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f = next(iter(train_dataloader))

            # Conditional NFs (marginalized nuisance) fitting
            nuisance_optimizer.zero_grad()

            # Representation -> Adding noise + concat of factual treatment -> Conditional distribution
            repr_f = self.repr_nn(cov_f)
            noised_out_f = out_f + self.noise_std_Y * torch.randn_like(out_f)
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
            context_f = torch.cat([noised_repr_f, treat_f], dim=1)
            log_prob_f = self.cond_flow_dist.condition(context_f).log_prob(noised_out_f)

            loss = - log_prob_f.mean()
            loss.backward()

            nuisance_optimizer.step()
            self.cond_flow_dist.clear_cache()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics({'train_cond_neg_log_prob': loss.item()}, step=step)

        train_dataloader = self.get_train_dataloader(cov_f_full, treat_f_full, out_f_full, self.target_batch_size)
        for step in tqdm(range(self.target_num_epochs)) if log else range(self.target_num_epochs):
            # NF (target) fitting
            if self.target_mode == 'batch':
                cov_f_full, treat_f_full, out_f_full = next(iter(train_dataloader))

            for i, treat_option in enumerate(self.treat_options):
                target_optimizers[i].zero_grad()

                # Cross-entropy based teaching
                cross_entropy = self.cross_entropy(self.flow_dist[i], treat_option, cov_f_full)
                cross_entropy.backward()

                target_optimizers[i].step()
                self.flow_dist[i].clear_cache()

                if step % 50 == 0 and log:
                    self.mlflow_logger.log_metrics({f'train_nce_{int(treat_option)}': cross_entropy.item()}, step=step)

            self.ema_target.update()

    # def cond_neg_cross_entropy(self, treat_f, out_f, cov_f):
    #     # Preparing data
    #     cov_f, _, _ = self.prepare_tensors(cov_f, treat_f, out_f, kind='torch')
    #     cov_f = torch.tensor(self.cov_scaler.transform(cov_f)).float()
    #
    #     neg_cross_entropy = 0.0
    #     with torch.no_grad():
    #         with self.ema_target.average_parameters():
    #             for i, treat_option in enumerate(self.treat_options):
    #                 neg_cross_entropy += - self.cross_entropy(self.flow_dist[i], treat_option, cov_f)
    #     return neg_cross_entropy

    def inter_target_log_prob(self, treat_pot, out_pot, with_grad=False) -> torch.Tensor:
        """
        Interventional log-probability
        @param with_grad: Enable gradient computation\
        @param treat_pot: Tensor of potential treatments
        @param out_pot: Tensor of potential outcome values
        @return: Tensor with log-probabilities
        """
        assert treat_pot.shape[0] == out_pot.shape[0]

        _, treat_pot, out_pot = self.prepare_tensors(None, treat_pot, out_pot, kind='torch')

        log_prob_pot = np.zeros((out_pot.shape[0], 1))

        ctx_mgr = torch.no_grad() if not with_grad else contextlib.suppress()
        with ctx_mgr:
            with self.ema_target.average_parameters():
                for i, treat_option in enumerate(self.treat_options):
                    if sum(treat_pot == treat_option) > 0:
                        log_prob_pot[treat_pot == treat_option] = \
                            self.flow_dist[i].log_prob(
                                out_pot[treat_pot.reshape(-1) == treat_option].reshape(-1, self.dim_out)).squeeze()

        return log_prob_pot

    def inter_log_prob(self, treat_pot, out_pot) -> torch.Tensor:
        """
        Interventional log-probability for large datasets
        @param treat_pot: Tensor of potential treatments
        @param out_pot: Tensor of potential outcome values
        @return: Tensor with log-probabilities
        """
        return self.inter_target_log_prob(treat_pot, out_pot)

    def inter_sample(self, treat_pot, n_samples) -> torch.Tensor:
        """
        Sampling from the estimated interventional density
        @param treat_pot: Tensor of potential treatments
        @param n_samples: number of samples
        @return: Tensor with sampled data
        """
        _, treat_pot, _ = self.prepare_tensors(None, treat_pot, None, kind='torch')
        sampled_out_pot = np.zeros((n_samples, treat_pot.shape[0], self.dim_out))

        with torch.no_grad():
            with self.ema_target.average_parameters():
                for i, treat_option in enumerate(self.treat_options):
                    if sum(treat_pot == treat_option) > 0:
                        sampled_out_pot[:, (treat_pot == treat_option).squeeze(), :] =\
                            self.flow_dist[i].sample((n_samples, len(treat_pot == treat_option)))
        return sampled_out_pot

    def inter_mean(self, treat_option, **kwargs) -> torch.Tensor:
        """
        Mean of potential outcomes
        @param treat_option: Treatment 0 / 1
        @return: mean
        """
        treat_pot = treat_option * np.ones((self.n_mc, ))
        return self.inter_sample(treat_pot, 1).mean((0, 1))


class AIPTWINFs(CovariateAdjustedINFs):
    """
    INFs (main): Fully-parametric A-IPTW interventional density estimator with (i) conditional normalizing flows and
    (ii) normalizing flows
    """

    tune_criterion = "cond_log_prob"

    def __init__(self, args: DictConfig = None, **kwargs):
        super(AIPTWINFs, self).__init__(args)

        # Model hyparams & Train params
        self.has_prop_score = True
        self.prop_alpha = args.model.prop_alpha
        self.clip_prop = args.model.clip_prop

        # Model parameters
        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid], param_dims=[self.dim_hid, self.dim_treat]).float()

    def fit(self, train_data_dict: dict, log: bool) -> None:
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        # Preparing data
        cov_f_full, treat_f_full, out_f_full = self.prepare_train_data(train_data_dict)
        cov_f_full = torch.tensor(self.cov_scaler.fit_transform(cov_f_full)).float()
        train_dataloader = self.get_train_dataloader(cov_f_full, treat_f_full, out_f_full, self.nuisance_batch_size)

        # Preparing optimizers
        nuisance_optimizer = self.get_nuisance_optimizer()
        target_optimizers, self.ema_target = self.get_target_optimizers()
        self.to(self.device)

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving train train_data_dict
        self.save_train_data_to_buffer(cov_f_full, treat_f_full, out_f_full)

        for step in tqdm(range(self.num_epochs)):
            cov_f, treat_f, out_f = next(iter(train_dataloader))

            # Conditional NFs (marginalized nuisance) fitting
            nuisance_optimizer.zero_grad()

            # Representation + Propensity score -> Adding noise + concat of factual treatment -> Conditional distribution
            repr_f, prop_preds = self.repr_nn(cov_f)
            noised_out_f = out_f + self.noise_std_Y * torch.randn_like(out_f)
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
            context_f = torch.cat([noised_repr_f, treat_f], dim=1)
            log_prob_f = self.cond_flow_dist.condition(context_f).log_prob(noised_out_f)
            bce_loss = torch.binary_cross_entropy_with_logits(prop_preds, treat_f).mean()

            loss = - log_prob_f.mean() + self.prop_alpha * bce_loss

            loss.backward()
            nuisance_optimizer.step()
            self.cond_flow_dist.clear_cache()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics({
                    'train_cond_neg_log_prob': - log_prob_f.mean(),
                    'train_bce_loss': bce_loss
                }, step=step)

        train_dataloader = self.get_train_dataloader(cov_f_full, treat_f_full, out_f_full, self.target_batch_size)
        for step in tqdm(range(self.target_num_epochs)):
            # NF (target) fitting
            if self.target_mode == 'batch':
                cov_f_full, treat_f_full, out_f_full = next(iter(train_dataloader))

            with torch.no_grad():
                _, prop_preds_full = self.repr_nn(cov_f_full)

            for i, treat_option in enumerate(self.treat_options):
                target_optimizers[i].zero_grad()

                # Cross-entropy + IPTW (bias correction) based teaching
                treat_pot = treat_option * torch.ones((self.target_nce_bins,))
                if self.dim_out == 1:  # Grid based approximation of cross-entropy
                    noised_out_pot, bin_size = self._get_out_pot_grid()
                    log_prob_pot = self.flow_dist[i].log_prob(noised_out_pot).squeeze()
                    with torch.no_grad():
                        target_log_prob_pot = \
                            self.inter_nuisances_log_prob(treat_pot, noised_out_pot, cov_f_batch=cov_f_full).squeeze()
                    if self.target_quadrature == 'rect':
                        cross_entropy = - (target_log_prob_pot.exp() * log_prob_pot).sum() * bin_size
                    elif self.target_quadrature == 'trap':
                        cross_entropy = - 0.5 * (target_log_prob_pot.exp() * log_prob_pot).unfold(0, 2, 1).sum() * bin_size
                    else:
                        raise NotImplementedError()

                    with torch.no_grad():
                        target_cond_log_prob_pot = self.inter_nuisances_log_prob(treat_pot, noised_out_pot, marginalize=False,
                                                                                 cov_f_batch=cov_f_full).squeeze()
                    if self.target_quadrature == 'rect':
                        cond_cross_entropy = - (target_cond_log_prob_pot.exp() * log_prob_pot.unsqueeze(-1)).sum(0) * bin_size
                    elif self.target_quadrature == 'trap':
                        cond_cross_entropy = - 0.5 * (target_cond_log_prob_pot.exp() *
                                                      log_prob_pot.unsqueeze(-1)).unfold(0, 2, 1).sum(0).sum(1) * bin_size
                    else:
                        raise NotImplementedError()
                else:  # MC sampling for approximation of cross-entropy
                    with torch.no_grad():
                        # out_pot_sample = self.inter_nuisances_sample(treat_pot, 1, cov_f_batch=cov_f_full).squeeze()
                        out_cond_pot_sample = self.inter_nuisances_sample(treat_pot, 1, cov_f_batch=cov_f_full,
                                                                          marginalize=False).squeeze()
                        out_pot_sample = \
                            torch.gather(out_cond_pot_sample, 1,
                                         torch.randint(cov_f_full.shape[0], (self.target_nce_bins, 1, self.dim_out))).squeeze(1)

                    cross_entropy = - self.flow_dist[i].log_prob(out_pot_sample).mean()
                    cond_cross_entropy = - self.flow_dist[i].log_prob(out_cond_pot_sample).mean(0)

                log_prob_f = self.flow_dist[i].log_prob(out_f_full).squeeze()

                prop = torch.sigmoid(prop_preds_full) if treat_option == 1.0 else 1.0 - torch.sigmoid(prop_preds_full)

                bias_correction = ((((treat_f_full == treat_option) & (prop >= self.clip_prop)) / (prop + 1e-9)).squeeze() *
                                   (- log_prob_f - cond_cross_entropy)).mean()
                loss = cross_entropy + bias_correction

                loss.backward()
                target_optimizers[i].step()
                self.flow_dist[i].clear_cache()

                if step % 50 == 0 and log:
                    self.mlflow_logger.log_metrics({
                        f'train_nce_{int(treat_option)}': cross_entropy.item(),
                        f'bias_correction_{int(treat_option)}': bias_correction.item()
                    }, step=step)

            self.ema_target.update()


class AIPTWCNFTruncatedSeries(PluginINFs):
    """
    CNF+TS: Fully-parametric A-IPTW interventional density estimator with (i) conditional normalizing flows and (ii) truncated
    series
    Kennedy, E. H., Balakrishnan, S., and Wasserman, L. Semi-parametric counterfactual density estimation. Biometrika, 2023.
    """

    tune_criterion = "cond_log_prob"

    def __init__(self, args: DictConfig = None, **kwargs):
        super(AIPTWCNFTruncatedSeries, self).__init__(args)

        # Model hyparams & Train params
        self.has_prop_score = True
        self.prop_alpha = args.model.prop_alpha
        self.clip_prop = args.model.clip_prop

        # Model hyparams & Train params
        self.noise_ce = args.model.noise_ce
        self.target_nce_bins = args.model.target_nce_bins
        self.norm_bins = self.target_nce_bins  # Renormalization
        self.target_quadrature = args.model.target_quadrature
        self.target_n_basis = args.model.target_n_basis

        # Model parameters
        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid], param_dims=[self.dim_hid, self.dim_treat]).float()
        self.betas = {}
        self.norm_const = np.array([1.0, 1.0])  # Will be updated after fit

    def orth_basis(self, out, cond_sample=False) -> torch.Tensor:
        """
        Orthogonal basis functions
        @param out: Outcome points to evaluate the orthogonal basis functions
        @param cond_sample: Flag whether out is a sample from the conditional ditribution, i.e., it has one extra dimension
        @return: Orthogonal basis functions evaluated at out
        """
        out_scaled = (out - self.out_f.min(0)[0]) / (self.out_f.max(0)[0] - self.out_f.min(0)[0])
        if self.dim_out == 1:
            j = torch.tensor(range(1, self.target_n_basis + 1), dtype=float).reshape(1, -1)
            return np.sqrt(2) * torch.cos(torch.pi * j * out_scaled)
        elif self.dim_out == 2:
            j = torch.tensor(range(1, self.target_n_basis + 1), dtype=float)
            i = torch.tensor(range(1, self.target_n_basis + 1), dtype=float)
            i, j = torch.meshgrid(i, j)
            i, j = i.unsqueeze(0), j.unsqueeze(0)
            if cond_sample:
                i, j = i.unsqueeze(0), j.unsqueeze(0)
            out_scaled = out_scaled.unsqueeze(-1).unsqueeze(-1)
            if not cond_sample:
                return 2 * torch.cos(torch.pi * j * out_scaled[:, 0]) * torch.cos(torch.pi * i * out_scaled[:, 1])
            else:
                return 2 * torch.cos(torch.pi * j * out_scaled[:, :, 0]) * torch.cos(torch.pi * i * out_scaled[:, :, 1])
        else:
            raise NotImplementedError()

    def fit(self, train_data_dict: dict, log: bool) -> None:
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        # Preparing data
        cov_f_full, treat_f_full, out_f_full = self.prepare_train_data(train_data_dict)
        cov_f_full = torch.tensor(self.cov_scaler.fit_transform(cov_f_full)).float()
        train_dataloader = self.get_train_dataloader(cov_f_full, treat_f_full, out_f_full, self.nuisance_batch_size)

        # Preparing optimizers
        nuisance_optimizer = self.get_nuisance_optimizer()
        self.to(self.device)

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving train train_data_dict
        self.save_train_data_to_buffer(cov_f_full, treat_f_full, out_f_full)

        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f = next(iter(train_dataloader))

            # Conditional NFs (marginalized nuisance) fitting
            nuisance_optimizer.zero_grad()

            # Representation + Propensity score -> Adding noise + concat of factual treatment -> Conditional distribution
            repr_f, prop_preds = self.repr_nn(cov_f)
            noised_out_f = out_f + self.noise_std_Y * torch.randn_like(out_f)
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
            context_f = torch.cat([noised_repr_f, treat_f], dim=1)
            log_prob_f = self.cond_flow_dist.condition(context_f).log_prob(noised_out_f)
            bce_loss = torch.binary_cross_entropy_with_logits(prop_preds, treat_f).mean()

            loss = - log_prob_f.mean() + self.prop_alpha * bce_loss

            loss.backward()
            nuisance_optimizer.step()
            self.cond_flow_dist.clear_cache()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics({
                    'train_cond_neg_log_prob': - log_prob_f.mean(),
                    'train_bce_loss': bce_loss
                }, step=step)

        with torch.no_grad():
            _, prop_preds_full = self.repr_nn(cov_f_full)

        b_y_f = self.orth_basis(out_f_full)

        for i, treat_option in enumerate(self.treat_options):
            prop = torch.sigmoid(prop_preds_full) if treat_option == 1.0 else 1.0 - torch.sigmoid(prop_preds_full)

            treat_pot = treat_option * torch.ones((self.target_nce_bins,))
            if self.dim_out == 1:  # Grid based approximation of cross-entropy
                noised_out_pot, bin_size = self._get_out_pot_grid()
                b_y_pot = self.orth_basis(noised_out_pot)

                with torch.no_grad():
                    target_cond_log_prob_pot = self.inter_nuisances_log_prob(treat_pot, noised_out_pot, marginalize=False,
                                                                             cov_f_batch=cov_f_full).squeeze()

                betas_cond = (target_cond_log_prob_pot.exp().unsqueeze(-1) * b_y_pot.unsqueeze(1)).sum(0) * bin_size
                betas_bias_corrected = ((treat_f_full == treat_option) & (prop >= self.clip_prop)) /\
                                       (prop + 1e-9) * (b_y_f - betas_cond) + betas_cond
                self.betas[treat_option] = betas_bias_corrected.mean(0)
            elif self.dim_out == 2:
                with torch.no_grad():
                    out_cond_pot_sample = \
                        self.inter_nuisances_sample(treat_pot, 1, cov_f_batch=cov_f_full, marginalize=False).squeeze()

                b_y_cond = self.orth_basis(out_cond_pot_sample, cond_sample=True)
                betas_cond = b_y_cond.mean(0)
                betas_bias_corrected = (((treat_f_full == treat_option) & (prop >= self.clip_prop)) /
                                        (prop + 1e-9)).unsqueeze(-1) * (b_y_f - betas_cond) + betas_cond
                self.betas[treat_option] = betas_bias_corrected.mean(0)
            else:
                raise NotImplementedError()  # Higher dimensions are computationally infeasible

        self.set_norm_consts()

    def inter_target_log_prob(self, treat_pot, out_pot, with_grad=False) -> torch.Tensor:
        """
        Interventional log-probability
        @param with_grad: Enable gradient computation\
        @param treat_pot: Tensor of potential treatments
        @param out_pot: Tensor of potential outcome values
        @return: Tensor with log-probabilities
        """
        assert treat_pot.shape[0] == out_pot.shape[0]

        _, treat_pot, out_pot = self.prepare_tensors(None, treat_pot, out_pot, kind='torch')

        log_prob_pot = np.zeros((out_pot.shape[0], 1))

        ctx_mgr = torch.no_grad() if not with_grad else contextlib.suppress()
        with ctx_mgr:
            for treat_option, norm_const in zip(self.treat_options, self.norm_const):
                if sum(treat_pot == treat_option) > 0:
                    norm_const = torch.tensor(norm_const)
                    b_y_pot = self.orth_basis(out_pot[treat_pot.reshape(-1) == treat_option].reshape(-1, self.dim_out))
                    if self.dim_out == 1:
                        prob_pot = 1.0 + (self.betas[treat_option].unsqueeze(0) * b_y_pot).sum(1)
                        prob_pot = prob_pot / (self.out_f.max(0)[0] - self.out_f.min(0)[0])
                    elif self.dim_out == 2:
                        prob_pot = 1.0 + (self.betas[treat_option].unsqueeze(0) * b_y_pot).sum((1, 2))
                        prob_pot = prob_pot / torch.prod(self.out_f.max(0)[0] - self.out_f.min(0)[0])

                    prob_pot = prob_pot / norm_const
                    # Zeroing negative values
                    prob_pot[(prob_pot <= 0.0)] = 1e-10
                    # Zeroing values outside of support
                    prob_pot[(out_pot < self.out_f.min(0)[0]).any(1) | (out_pot > self.out_f.max(0)[0]).any(1)] = 1e-10
                    log_prob_pot[treat_pot == treat_option] = prob_pot.log()

        return log_prob_pot

    def inter_log_prob(self, treat_pot, out_pot) -> torch.Tensor:
        """
        Interventional log-probability for large datasets
        @param treat_pot: Tensor of potential treatments
        @param out_pot: Tensor of potential outcome values
        @return: Tensor with log-probabilities
        """
        return self.inter_target_log_prob(treat_pot, out_pot)
