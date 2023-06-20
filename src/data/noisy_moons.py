import numpy as np
from sklearn.datasets import make_moons
from sklearn import preprocessing


class NoisyMoons:
    def __init__(self, seed=42, n_samples=5000, noise=0.5, normalize_out=True, theta_0=np.pi / 4, theta_1=-np.pi / 4, **kwargs):

        self.n_samples = n_samples
        self.seed = seed
        self.noise = noise

        self.normalize_out = normalize_out
        self.out_scaler = preprocessing.StandardScaler()

        self.theta_0 = theta_0
        self.theta_1 = theta_1

    def get_data(self) -> dict:
        np.random.seed(self.seed)

        # Sampling random rotations
        noise = 0.1 * np.random.randn(self.n_samples)
        theta_0, theta_1 = self.theta_0 + noise, self.theta_1 + noise
        rot_0 = np.array([[np.cos(theta_0), -np.sin(theta_0)], [np.sin(theta_0), np.cos(theta_0)]])
        rot_1 = np.array([[np.cos(theta_1), -np.sin(theta_1)], [np.sin(theta_1), np.cos(theta_1)]])

        X, T = make_moons(self.n_samples, noise=self.noise, random_state=self.seed)
        Y = np.empty_like(X)
        Y0_pot = np.empty_like(X)
        Y1_pot = np.empty_like(X)
        for i, t in enumerate(T):
            Y0_pot[i, :] = X[i, :] @ rot_0[:, :, i] + noise[i]
            Y1_pot[i, :] = X[i, :] @ rot_1[:, :, i] + noise[i]
            if t == 0:
                Y[i, :] = X[i, :] @ rot_0[:, :, i] + noise[i]
            else:
                Y[i, :] = X[i, :] @ rot_1[:, :, i] + noise[i]

        if self.normalize_out:
            self.out_scaler.fit(np.concatenate([Y0_pot, Y1_pot], 0).reshape(-1, 2))
            Y = self.out_scaler.transform(Y)
            Y0_pot = self.out_scaler.transform(Y0_pot)
            Y1_pot = self.out_scaler.transform(Y1_pot)

        return {
            'cov_f': X,
            'treat_f': T.astype(float).reshape(-1, 1),
            'out_f': Y,
            'out_pot_0': Y0_pot,
            'out_pot_1': Y1_pot
        }
