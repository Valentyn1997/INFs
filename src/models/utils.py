import numpy as np
from sklearn.gaussian_process.kernels import RBF
import pyro.distributions.transforms as T
from pyro.distributions.torch_transform import TransformModule


class Affine(TransformModule):
    """AffineTransform with trainable parameters"""
    def __init__(self, input_dim):
        super().__init__(cache_size=1)
        self.input_dim = input_dim
        self.transform = T.AffineTransform(self.loc, self.scale)

    def _call(self, x):
        return self.transform._call(x)

    def _inverse(self, y):
        return self.transform._inverse(y)

    def log_abs_det_jacobian(self, x, y):
        return self.transform.log_abs_det_jacobian(x, y)

    @property
    def sign(self):
        return self.transform.sign


def subset_by_indices(data_dict: dict, indices: list):
    """
    Subset of the data dictionary
    @param data_dict: data dictionary
    @param indices: indices for a subset
    @return: new data dictionary
    """
    subset_data_dict = {}
    for (k, v) in data_dict.items():
        subset_data_dict[k] = np.copy(data_dict[k][indices])
    return subset_data_dict


class NormalizedRBF:
    """
    Normalized radial basis function, used for KDE and DKME
    """
    def __init__(self, sd):
        self.sd = sd
        self.rbf = RBF(np.sqrt(self.sd / 2))

    def __call__(self, x1, x2):
        return 1 / np.sqrt(np.pi * self.sd) * self.rbf(x1, x2)
