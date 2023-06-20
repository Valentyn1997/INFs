import torch
import pyro.distributions as dist
from omegaconf import DictConfig
import logging
from pyro.nn import DenseNN
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from torch.distributions import Normal, OneHotCategorical, MixtureSameFamily, Independent, MultivariateNormal


from src.models import InterventionalDensityEstimator

logger = logging.getLogger(__name__)


class PluginMarginalizedTeacher(InterventionalDensityEstimator):
    """
    Semi-parametric model for interventional density based on abstract conditional density estimator (marginalized teacher)
    """
    tune_criterion = 'cond_log_prob'

    def __init__(self, args: DictConfig = None, **kwargs):
        super(PluginMarginalizedTeacher, self).__init__(args)

        self.cov_scaler = StandardScaler()

        # Model hyparams & Train params
        self.teacher_hid_dim_multiplier = args.model.teacher_hid_dim_multiplier
        self.noise_std_X, self.noise_std_Y = args.model.noise_std_X, args.model.noise_std_Y
        self.num_epochs = args.model.num_epochs
        self.teacher_lr = args.model.teacher_lr
        self.batch_size = args.model.batch_size
        self.n_mc = args.model.n_mc  # For the MC mean estimation

        self.dim_hid = self.teacher_hid_dim_multiplier
        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid], param_dims=[self.dim_hid]).float()
        self.cond_dist_nn = None

    def _cond_log_prob(self, context, out):
        raise NotImplementedError()

    def _cond_dist(self, context):
        raise NotImplementedError()

    def get_train_dataloader(self, cov_f, treat_f, out_f):
        training_data = TensorDataset(cov_f, treat_f, out_f)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
        return train_dataloader

    def prepare_train_data(self, data_dict: dict):
        cov_f, treat_f, out_f = self.prepare_tensors(data_dict['cov_f'], data_dict['treat_f'], data_dict['out_f'], kind='torch')
        self.hparams.dataset.n_samples_train = cov_f.shape[0]
        return cov_f, treat_f, out_f

    def get_teacher_optimizer(self):
        modules = torch.nn.ModuleList([self.repr_nn, self.cond_dist_nn])
        return torch.optim.SGD(list(modules.parameters()), lr=self.teacher_lr, momentum=0.9)

    def fit(self, train_data_dict: dict, log: bool):
        # Preparing data
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        cov_f = torch.tensor(self.cov_scaler.fit_transform(cov_f)).float()
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f)

        # Preparing optimizers
        teacher_optimizer = self.get_teacher_optimizer()

        # Logging
        self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        # Saving train train_data_dict
        self.save_train_data_to_buffer(cov_f, treat_f, out_f)

        # Conditional NFs (marginalized teacher) fitting
        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f = next(iter(train_dataloader))
            teacher_optimizer.zero_grad()

            # Representation -> Adding noise + concat of factual treatment -> Conditional distribution
            repr_f = self.repr_nn(cov_f)
            noised_out_f = out_f + self.noise_std_Y * torch.randn_like(out_f)
            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
            context = torch.cat([noised_repr_f, treat_f], dim=1)
            log_prob = self._cond_log_prob(context, noised_out_f)

            loss = - log_prob.mean()
            loss.backward()

            teacher_optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics({'train_cond_neg_log_prob': loss.item()}, step=step)

    def inter_log_prob(self, treat_pot, out_pot):
        if out_pot.shape[0] > 25000 or self.cov_f.shape[0] > 25000:
            log_prob_pot = torch.zeros((out_pot.shape[0], 1))
            batch_size = 1000
            for i in range((out_pot.shape[0] - 1) // batch_size + 1):
                log_prob_pot[i * batch_size: (i + 1) * batch_size] = \
                    self.inter_teachers_log_prob(treat_pot[i * batch_size: (i + 1) * batch_size],
                                                 out_pot[i * batch_size: (i + 1) * batch_size])
            return log_prob_pot
        else:
            return self.inter_teachers_log_prob(treat_pot, out_pot)

    def inter_teachers_log_prob(self, treat_pot, out_pot):
        assert treat_pot.shape[0] == out_pot.shape[0]
        n = out_pot.shape[0]
        _, treat_pot, out_pot = self.prepare_tensors(None, treat_pot, out_pot, kind='torch')

        # Exploding train_data_dict for inference
        cov_f = self.cov_f.repeat(n, 1, 1)
        treat_pot = treat_pot.unsqueeze(1).repeat(1, cov_f.shape[1], 1)
        out_pot = out_pot.unsqueeze(1).repeat(1, cov_f.shape[1], 1)

        with torch.no_grad():
            # Propensity is not used
            repr_f = self.repr_nn(cov_f)
            context = torch.cat([repr_f, treat_pot], dim=2)

            log_prob_pot = self._cond_log_prob(context, out_pot)
            log_prob_pot = log_prob_pot.exp().mean(1, keepdim=True).log()

        return log_prob_pot

    def inter_sample(self, treat_pot, n_samples):
        n, cov_n = treat_pot.shape[0], self.cov_f.shape[0]
        _, treat_pot, _ = self.prepare_tensors(None, treat_pot, None, kind='torch')
        cov_f = self.cov_f.repeat(n, 1, 1)
        treat_pot = treat_pot.unsqueeze(1).repeat(1, cov_f.shape[1], 1)

        with torch.no_grad():
            repr_f = self.repr_nn(cov_f)
            context = torch.cat([repr_f, treat_pot], dim=2)
            cond_dist = self._cond_dist(context)
            sampled_out_cond = cond_dist.sample((n_samples,))
            if len(sampled_out_cond.shape) == 3:
                sampled_out_cond = sampled_out_cond.unsqueeze(-1)
            sampled_out_pot = torch.gather(sampled_out_cond, 2, torch.randint(cov_n, (n_samples, n, 1, self.dim_out))).squeeze(2)

        return sampled_out_pot

    def cond_log_prob(self, treat_f, out_f, cov_f):
        # Preparing data
        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, treat_f, out_f, kind='torch')
        cov_f = torch.tensor(self.cov_scaler.transform(cov_f)).float()

        with torch.no_grad():
            repr = self.repr_nn(cov_f)
            context = torch.cat([repr, treat_f], dim=1)
            log_prob = self._cond_log_prob(context, out_f)
        return log_prob

    def inter_mean(self, treat_option, **kwargs):
        treat_pot = treat_option * torch.ones((self.cov_f.shape[0], 1)).type_as(self.cov_f)

        with torch.no_grad():
            # Propensity is not used
            repr_f = self.repr_nn(self.cov_f)
            context = torch.cat([repr_f, treat_pot], dim=1)
            cond_dist = self._cond_dist(context)
            return cond_dist.mean.mean(0).numpy()


class PluginGaussianTARNetMarginalizedTeacher(PluginMarginalizedTeacher):
    """
    Semi-parametric model for interventional density based on TARNet with Gaussian cond outcomes (marginalized teacher)
    """
    def __init__(self, args: DictConfig = None, **kwargs):
        super(PluginGaussianTARNetMarginalizedTeacher, self).__init__(args)
        self.cond_dist_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                    param_dims=[self.dim_out]).float()
        self.log_std = torch.nn.Parameter(torch.zeros((self.dim_out, )))

    def get_teacher_optimizer(self):
        modules = torch.nn.ModuleList([self.repr_nn, self.cond_dist_nn])
        return torch.optim.SGD(list(modules.parameters()) + [self.log_std], lr=self.teacher_lr, momentum=0.9)

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict):
        model_args.noise_std_X = new_model_args['noise_std_X']
        model_args.noise_std_Y = new_model_args['noise_std_Y']
        model_args.teacher_lr = new_model_args['teacher_lr']
        model_args.batch_size = new_model_args['batch_size']

    def _cond_dist(self, context):
        mu = self.cond_dist_nn(context)
        if self.dim_out == 1:
            return Normal(mu, self.log_std.exp())
        else:
            return MultivariateNormal(mu, torch.diag(self.log_std.exp()))

    def _cond_log_prob(self, context, out):
        cond_normal = self._cond_dist(context)
        return cond_normal.log_prob(out).squeeze()


class PluginMixtureDensityMarginalizedTeacher(PluginMarginalizedTeacher):
    """
    Semi-parametric model for interventional density based on mixture density networks (marginalized teacher)
    """
    def __init__(self, args: DictConfig = None, **kwargs):
        super(PluginMixtureDensityMarginalizedTeacher, self).__init__(args)

        # Model hyparams & Train params
        self.teacher_num_comp = args.model.teacher_num_comp

        self.cond_dist_nn = DenseNN(self.dim_hid + self.dim_treat, [self.dim_hid],
                                    param_dims=[self.dim_out * self.teacher_num_comp,
                                                self.dim_out * self.teacher_num_comp,
                                                self.teacher_num_comp]).float()

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict):
        model_args.teacher_num_comp = new_model_args['teacher_num_comp']
        model_args.noise_std_X = new_model_args['noise_std_X']
        model_args.noise_std_Y = new_model_args['noise_std_Y']
        model_args.teacher_lr = new_model_args['teacher_lr']
        model_args.batch_size = new_model_args['batch_size']

    def cond_mixture(self, context):
        mu, log_std, logits = self.cond_dist_nn(context)
        cat_dist = OneHotCategorical(logits=logits)
        std = torch.exp(log_std).reshape(*log_std.shape[:-1], self.teacher_num_comp, self.dim_out)
        mu = mu.reshape(*mu.shape[:-1], self.teacher_num_comp, self.dim_out)
        norm_comps = MultivariateNormal(mu, torch.diag_embed(std))
        return cat_dist, norm_comps

    def _cond_log_prob(self, context, out):
        cat_dist, norm_comps = self.cond_mixture(context)
        return torch.logsumexp(cat_dist.probs.log() + norm_comps.log_prob(out.unsqueeze(-2)), dim=-1)

    def _cond_dist(self, context):
        cat_dist, norm_comps = self.cond_mixture(context)
        return MixtureSameFamily(cat_dist._categorical, Independent(norm_comps, 0))
