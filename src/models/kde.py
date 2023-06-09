import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
from pyro.nn import DenseNN
from torch.utils.data import TensorDataset, DataLoader
from src.models import InterventionalDensityEstimator
from omegaconf import DictConfig
from scipy.special import expit
from typing import Tuple

from src.models.utils import NormalizedRBF

logger = logging.getLogger(__name__)


class AIPTWKDE(InterventionalDensityEstimator):
    """
    KDE: Semi-parametric A-IPTW interventional density estimator with kernels
    Kim, K., Kim, J., and Kennedy, E. H. Causal effects based on distributional distances. arXiv preprint arXiv:1806.02935, 2018.
    """

    tune_criterion = 'neg_mse_and_neg_bce'

    def __init__(self, args: DictConfig = None, **kwargs):
        super(AIPTWKDE, self).__init__(args)
        self.cov_scaler = StandardScaler()

        # assert self.dim_out == 1

        # Model hyperparamenters & Train params
        self.sd_y = args.model.sd_y
        self.pure_functional = args.model.pure_functional
        self.normalized = args.model.normalized
        self.norm_bins = args.model.norm_bins
        self.norm_const = np.array([1.0, 1.0])  # Will be updated after fit
        self.clip_prop = args.model.clip_prop
        self.dim_hid = args.model.hid_dim_multiplier
        self.num_epochs = args.model.num_epochs
        self.lr = args.model.lr
        self.prop_alpha = args.model.prop_alpha
        self.batch_size = args.model.batch_size

        # Model paramenters
        self.normalized_rbf_y = {treat_option: NormalizedRBF(self.sd_y) for treat_option in self.treat_options}

        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid], param_dims=[self.dim_hid, self.dim_treat]).float()
        if not self.pure_functional:
            self.regression_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid], param_dims=[self.dim_out]).float()
        else:
            self.regression_nn = DenseNN(self.dim_hid + self.dim_treat + self.dim_out, [self.dim_hid],
                                         param_dims=[self.dim_out]).float()

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict) -> None:
        """
        Set hyperparameters during tuning
        @param model_args: Default hparameters
        @param new_model_args: Hparameters for tuning
        """
        model_args.lr = new_model_args['lr']
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

    def get_train_dataloader(self, cov_f, treat_f, out_f) -> DataLoader:
        training_data = TensorDataset(cov_f, treat_f, out_f)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
        return train_dataloader

    def forward_nuance(self, cov_f, out_f, treat_f) -> Tuple[torch.Tensor]:
        """
        Forward pass for the nuance network
        @param cov_f: Tensor with factual covariates
        @param out_f: Tensor with factual outcomes
        @param treat_f: Tensor with factual treatments
        @return: Tensors with mse_loss, bce_loss
        """
        repr_f, prop_preds = self.repr_nn(cov_f)
        if not self.pure_functional:
            repr_treat_f = torch.cat([repr_f, treat_f], dim=1)
            out_pred = self.regression_nn(repr_treat_f)
            mse_loss = (out_pred - out_f) ** 2
        else:
            out_pot = torch.rand_like(out_f) * (out_f.min() - out_f.max()) + out_f.max()
            repr_treat_f = torch.cat([repr_f, treat_f], dim=1)
            repr_treat_f_out_pot = torch.cat([repr_treat_f.unsqueeze(1).expand(repr_treat_f.shape[0],
                                                                               out_pot.shape[0],
                                                                               repr_treat_f.shape[1]),
                                              out_pot.unsqueeze(0).repeat(repr_treat_f.shape[0], 1, 1)], 2)
            mu_pred = self.regression_nn(repr_treat_f_out_pot).squeeze()
            mu = torch.tensor(self.normalized_rbf_y(out_f, out_pot)).type_as(mu_pred)

            mse_loss = ((mu_pred - mu) ** 2).mean(1)

        bce_loss = torch.binary_cross_entropy_with_logits(prop_preds, treat_f)
        return mse_loss, bce_loss

    def fit_nuacance_models(self, cov_f, out_f, treat_f, log: bool) -> None:
        """
        Fitting the nuisance network of the estimator
        @param cov_f: Tensor with factual covariates
        @param out_f: Tensor with factual outcomes
        @param treat_f: Tensor with factual treatments
        @param log: Logging to the MlFlow
        """
        modules = torch.nn.ModuleList([self.repr_nn, self.regression_nn])
        optimizer = torch.optim.Adam(modules.parameters(), lr=self.lr)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f)

        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f = next(iter(train_dataloader))
            optimizer.zero_grad()
            mse_loss, bce_loss = self.forward_nuance(cov_f, out_f, treat_f)
            loss = mse_loss.mean() + self.prop_alpha * bce_loss.mean()
            loss.backward()
            optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics({'train_mse_loss': mse_loss.mean(), 'train_bce_loss': bce_loss.mean()}, step=step)

    def get_nuacance_predictions(self, cov, treat) -> np.array:
        """
        Get prediction of the nuisance network
        @param cov: Tensor of covariates
        @param treat: Tensor of treatments
        @return: Tensor of the predicted outcomes
        """
        with torch.no_grad():
            repr, prop_preds = self.repr_nn(cov)
            if not self.pure_functional:
                repr_treat = torch.cat([repr, treat], dim=1)
                out_pred = self.regression_nn(repr_treat)
                return prop_preds.numpy(), out_pred.numpy()
            else:
                repr_treat = torch.cat([repr, treat], dim=1)

                def normalized_rbf_pred(out_pot):
                    with torch.no_grad():
                        out_pot = torch.tensor(out_pot).reshape(-1, self.dim_out).float()
                        repr_treat_f_out_pot = torch.cat([repr_treat.unsqueeze(1).expand(repr_treat.shape[0], out_pot.shape[0],
                                                                                         repr_treat.shape[1]),
                                                          out_pot.unsqueeze(0).repeat(repr_treat.shape[0], 1, 1)], 2)
                        mu_pred = self.regression_nn(repr_treat_f_out_pot).squeeze()
                    return mu_pred.numpy()

                return prop_preds.numpy(), normalized_rbf_pred

    # def clip_quantiles(self, prop_preds):
    #     return np.clip(prop_preds, np.nanquantile(prop_preds, self.clip_prop_quantile), 1.0)

    def fit(self, train_data_dict: dict, log: bool) -> None:
        """
        Fitting the estimator
        @param train_data_dict: Training data dictionary
        @param log: Logging to the MlFlow
        """
        # Preparing data
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        cov_f = torch.tensor(self.cov_scaler.fit_transform(cov_f)).float()

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving sample for non-parametric estimation
        self.save_train_data_to_buffer(cov_f, treat_f, out_f)

        # Median heuristic for sd_y
        self.set_sd_y_median_heuristic() if self.hparams.model.median_sd_y_heuristic else None

        # S-learner fitting + propensity score
        self.fit_nuacance_models(cov_f, out_f, treat_f, log)

        # Saving predictions of rbf functionals + Propensity scores
        prop, self.out_f_pred = self.get_nuacance_predictions(cov_f, treat_f)
        self.out_pot_pred = {}
        for treat_option in self.treat_options:
            treat_pot = treat_option * torch.ones((cov_f.shape[0], 1)).float()
            _, self.out_pot_pred[treat_option] = self.get_nuacance_predictions(cov_f, treat_pot)
        self.prop = expit(prop)

        # Calculating normalization constants for correct log-likelihood
        self.set_norm_consts() if self.normalized else None

    def inter_log_prob(self, treat_pot, out_pot) -> np.array:
        """
        Interventional log-probability for large datasets
        @param treat_pot: Tensor of potential treatments
        @param out_pot: Tensor of potential outcome values
        @return: Tensor with log-probabilities
        """
        _, treat_pot, out_pot = self.prepare_tensors(None, treat_pot, out_pot, kind='torch')
        prob_pot = np.zeros((out_pot.shape[0], 1))

        for treat_option, norm_const in zip(self.treat_options, self.norm_const):
            if not self.pure_functional:
                mu_a = self.normalized_rbf_y[treat_option](self.out_pot_pred[treat_option],
                                                           out_pot[treat_pot.reshape(-1) == treat_option])
                mu_A = self.normalized_rbf_y[treat_option](self.out_f_pred, out_pot[treat_pot.reshape(-1) == treat_option])
            else:
                mu_a = self.out_pot_pred[treat_option](out_pot[treat_pot.reshape(-1) == treat_option])
                mu_A = self.out_f_pred(out_pot[treat_pot.reshape(-1) == treat_option])
            T_Y = self.normalized_rbf_y[treat_option](self.out_f, out_pot[treat_pot.reshape(-1) == treat_option])

            if sum(treat_pot == treat_option) > 0:
                prop = 1.0 - self.prop if treat_option == 0.0 else self.prop
                bias_correction = (((self.treat_f.numpy() == treat_option) & (prop >= self.clip_prop)) / (prop + 1e-9) *
                                   (T_Y - mu_A))
                prob_pot[treat_pot == treat_option] = (mu_a.mean(0) + bias_correction.mean(0)) / norm_const

        prob_pot[prob_pot <= 0.0] = 1e-10  # Zeroing negative values
        return np.log(prob_pot)

    def neg_mse_and_neg_bce(self, treat_f, out_f, cov_f) -> torch.Tensor:
        """
        Sum of negative MSE and negative BCE
        @param treat_f: Tensor with factual treatments
        @param out_f: Tensor with factual outcomes
        @param cov_f: Tensor with factual covariates
        @return: Tensor with the sum of negative MSE and negative BCE
        """
        # Preparing data
        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, treat_f, out_f, kind='torch')
        cov_f = torch.tensor(self.cov_scaler.transform(cov_f)).float()

        with torch.no_grad():
            mse_loss, bce_loss = self.forward_nuance(cov_f, out_f, treat_f)

        return - mse_loss - self.prop_alpha * bce_loss

    def inter_mean(self, treat_option, **kwargs) -> float:
        """
        Mean of potential outcomes
        @param treat_option: Treatment 0 / 1
        @return: mean
        """
        logger.warning('Calculation of mean is not implemented for this method.')
        return 0.0
