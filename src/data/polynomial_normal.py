import numpy as np
from numpy.polynomial.polynomial import polyval
from sklearn import preprocessing


class PolynomialNormal:
    """
    Synthetic data using the SCM
    """

    def __init__(self, seed=42, n_samples=5000, cov_shift=3.0, C0=[1.2, 2.3, 0.25], C1=[1.5, 1.0, 0.1], normalize_out=False,
                 **kwargs):
        self.seed = seed
        self.n_samples_per_treat = n_samples // 2
        self.cov_shift = cov_shift

        self.scaler = 1  # 0.25
        self.normalize_out = normalize_out
        self.out_scaler = preprocessing.StandardScaler()

        self.C0 = self.scaler * np.array(C0)
        self.C1 = self.scaler * np.array(C1)

    def get_data(self) -> dict:
        np.random.seed(self.seed)

        X0 = self.cov_shift + np.random.randn(self.n_samples_per_treat, 1)
        X1 = np.random.randn(self.n_samples_per_treat, 1)

        noise_0 = self.scaler * np.random.randn(len(X0), 1)
        noise_1 = self.scaler * np.random.randn(len(X1), 1)

        Y0 = polyval(X0, self.C0) + noise_0
        Y1 = polyval(X1, self.C1) + noise_1

        Y0_cf = polyval(X0, self.C1) + noise_1
        Y1_cf = polyval(X1, self.C0) + noise_0

        Y0_pot = np.concatenate([Y0, polyval(X1, self.C0) + noise_1])
        Y1_pot = np.concatenate([polyval(X0, self.C1) + noise_0, Y1])

        out_f = np.concatenate([Y0, Y1], axis=0)
        out_cf = np.concatenate([Y0_cf, Y1_cf], axis=0)

        if self.normalize_out:
            self.out_scaler.fit(np.concatenate([Y0_pot, Y1_pot]).reshape(-1, 1))
            out_f = self.out_scaler.transform(out_f)
            out_cf = self.out_scaler.transform(out_cf)
            Y0_pot = self.out_scaler.transform(Y0_pot)
            Y1_pot = self.out_scaler.transform(Y1_pot)

        return {
            'cov_f': np.concatenate([X0, X1], axis=0),
            'treat_f': np.concatenate([np.zeros((self.n_samples_per_treat,)), np.ones((self.n_samples_per_treat,))]),
            'out_f': out_f,
            'out_cf': out_cf,
            'out_pot_0': Y0_pot,
            'out_pot_1': Y1_pot
        }
