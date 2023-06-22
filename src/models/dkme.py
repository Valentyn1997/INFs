import logging
import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from src.models import InterventionalDensityEstimator

from src.models.utils import NormalizedRBF

logger = logging.getLogger(__name__)


class PluginDKME(InterventionalDensityEstimator):
    """
    DKME: Non-parametric plugin interventional density estimator with distributional kernel mean embedding
    Muandet, K., Kanagawa, M., Saengkyongam, S., and Marukatat, S. Counterfactual mean embeddings. Journal of Machine Learning
    Research, 22:1–71, 2021.
    """

    tune_criterion = 'kernel_ridge_reg_neg_mse'

    def __init__(self, args: DictConfig = None, **kwargs):

        super(PluginDKME, self).__init__(args)

        self.cov_scaler = StandardScaler()

        # Model hyparams
        self.sd_y = {treat_option: args.model.sd_y for treat_option in self.treat_options}
        self.sd_x = args.model.sd_x
        self.eps = args.model.eps
        self.median_sd_y_heuristic = args.model.median_sd_y_heuristic
        self.norm_bins = args.model.norm_bins
        self.normalized = args.model.normalized

        # Model parameters
        self.normalized_rbf_y = \
            {treat_option: NormalizedRBF(np.sqrt(self.sd_y[treat_option] / 2)) for treat_option in self.treat_options}
        self.rbf_x = RBF(np.sqrt(self.sd_x / 2))
        self.betas = {}
        self.norm_const = np.array([1.0, 1.0])  # Will be updated after fit

    def prepare_train_data(self, train_data_dict: dict):
        cov_f, treat_f, out_f = self.prepare_tensors(train_data_dict['cov_f'], train_data_dict['treat_f'],
                                                     train_data_dict['out_f'], kind='numpy')
        self.hparams.dataset.n_samples_train = cov_f.shape[0]
        return cov_f, treat_f, out_f

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict):
        model_args.sd_x = new_model_args['sd_x']
        model_args.eps = new_model_args['eps']

    def fit(self, train_data_dict: dict, log: bool):
        # Preparing data
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        cov_f = self.cov_scaler.fit_transform(cov_f)

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving sample for non-parametric estimation
        self.save_train_data_to_buffer(cov_f, treat_f, out_f)

        # Median heuristic for sd_y
        self.set_sd_y_median_heuristic() if self.hparams.model.median_sd_y_heuristic else None

        # T-learners fitting
        if not log:
            return

        treat_options = np.unique(treat_f)
        self.betas = {}
        for treat_option in treat_options:
            K = self.rbf_x(cov_f[treat_f == treat_option], cov_f[treat_f == treat_option])
            K_tilde = self.rbf_x(cov_f[treat_f == treat_option], cov_f)

            n_cond, n_tot = cov_f[treat_f == treat_option].shape[0], cov_f.shape[0]
            self.betas[treat_option] = np.dot(np.dot(np.linalg.inv(K + n_cond * self.eps * np.eye(n_cond)), K_tilde),
                                              np.ones((n_tot, 1)) / n_tot)

        # Calculating normalization constants for correct log-likelihood
        self.set_norm_consts() if self.normalized else None

    def kernel_ridge_reg_neg_mse(self, treat_f, out_f, cov_f):
        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, treat_f, out_f, kind='numpy')
        cov_f = self.cov_scaler.transform(cov_f)

        mses = np.zeros_like(out_f)
        for treat_option in self.treat_options:
            ker_ridge_reg = GaussianProcessRegressor(alpha=self.eps, kernel=self.rbf_x, optimizer=None)
            ker_ridge_reg.fit(self.cov_f[self.treat_f == treat_option], self.out_f[self.treat_f == treat_option])
            out_pred = ker_ridge_reg.predict(cov_f[treat_f == treat_option]).reshape(-1, self.dim_out)
            mses[treat_f == treat_option] = ((out_pred - out_f[treat_f == treat_option]) ** 2)
        return - mses

    def inter_log_prob(self, treat_pot, out_pot):
        _, treat_pot, out_pot = self.prepare_tensors(None, treat_pot, out_pot, kind='numpy')
        prob_pot = np.zeros((out_pot.shape[0], 1))

        for treat_option, norm_const in zip(self.treat_options, self.norm_const):
            L = self.normalized_rbf_y[treat_option](self.out_f[self.treat_f == treat_option], out_pot[treat_pot == treat_option])
            betas = self.betas[treat_option]

            prob_pot[treat_pot == treat_option] = np.dot(betas.T, L).T / np.abs(betas).sum() / norm_const

        prob_pot[prob_pot <= 0.0] = 1e-10  # Zeroing negative values
        return np.log(prob_pot)

    def inter_mean(self, treat_option, **kwargs):
        L = self.out_f[self.treat_f == treat_option]
        betas = self.betas[treat_option]
        return (np.dot(betas.T, L) / np.abs(betas).sum()).squeeze()
