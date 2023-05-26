import numpy as np
import pandas as pd
from sklearn import preprocessing

from src import ROOT_PATH


class IHDP:
    def __init__(self, normalize_out=False, **kwargs):
        self.data_path = f"{ROOT_PATH}/data/ihdp_npci_1.csv"
        self.normalize_out = normalize_out
        self.out_scaler = preprocessing.StandardScaler()

    def get_data(self):
        data = pd.read_csv(self.data_path, header=None)
        col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"]
        covarites = []
        for i in range(1, 26):
            col.append("x" + str(i))
            covarites.append("x" + str(i))
        data.columns = col
        data = data.astype({"treatment": 'bool'}, copy=False)

        Y0_pot = np.where(~data.treatment.values, data.y_factual.values, data.y_cfactual.values)
        Y1_pot = np.where(data.treatment.values, data.y_factual.values, data.y_cfactual.values)

        out_f = data.y_factual.values

        if self.normalize_out:
            self.out_scaler.fit(np.concatenate([Y0_pot, Y1_pot]).reshape(-1, 1))
            out_f = self.out_scaler.transform(out_f.reshape(-1, 1))

            Y0_pot = self.out_scaler.transform(Y0_pot.reshape(-1, 1))
            Y1_pot = self.out_scaler.transform(Y1_pot.reshape(-1, 1))

        return {
            'cov_f': data[covarites].values,
            'treat_f': data.treatment.astype(float).values,
            'out_f': out_f,
            'out_pot_0': Y0_pot,
            'out_pot_1': Y1_pot
        }
