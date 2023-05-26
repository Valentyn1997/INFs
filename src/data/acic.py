import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing

from src import ROOT_PATH


class ACIC:

    def __init__(self, year=2018, **kwargs):
        self.year = year
        if self.year == 2018:
            self.cov_data_path = f'{ROOT_PATH}/data/acic_2018/x.csv'
            self.out_data_path = f'{ROOT_PATH}/data/acic_2018/scaling'
        elif self.year == 2016:
            self.cov_data_path = f'{ROOT_PATH}/data/acic_2016_full/x.csv'
            self.out_data_path = f'{ROOT_PATH}/data/acic_2016_full/synth_outcomes'

        self.out_scaler = preprocessing.StandardScaler()

        self.simulation_files = sorted(glob.glob(f'{self.out_data_path}/*'))
        if self.year == 2018:
            self.simulation_files_f = [file for file in self.simulation_files if not file.endswith("_cf.csv")]
            self.simulation_files_cf = [file for file in self.simulation_files if file.endswith("_cf.csv")]

    def _postprocess_data(self, x, t, y_f, y_cf, dataset, standardize):
        if standardize:
            cov_scalar = preprocessing.StandardScaler()
            x = cov_scalar.fit_transform(x)
            self.out_scaler.fit(np.concatenate([dataset['y0'].values, dataset['y1'].values]).reshape(-1, 1))
            y_f = self.out_scaler.transform(y_f)
            y_cf = self.out_scaler.transform(y_cf)
        y0_pot = np.where(t == 0, y_f, y_cf)
        y1_pot = np.where(t == 1, y_f, y_cf)

        return {
            'cov_f': x,
            'treat_f': t,
            'out_f': y_f,
            'out_pot_0': y0_pot,
            'out_pot_1': y1_pot
        }

    def load_treatment_and_outcome_f_cf(self, covariates, file_f, file_cf, standardize=True):
        out_f = pd.read_csv(file_f, index_col='sample_id', header=0, sep=',')
        out_cf = pd.read_csv(file_cf, index_col='sample_id', header=0, sep=',')
        dataset = covariates.join(out_f, how='inner').join(out_cf, how='inner')
        t = dataset['z'].values
        y_f = dataset['y'].values
        y_cf = np.where(t == 0, dataset['y1'].values, dataset['y0'].values)
        x = dataset.values[:, :-4]
        t, y_f, y_cf, x = t.reshape(-1, 1).astype(float), y_f.reshape(-1, 1), y_cf.reshape(-1, 1), x
        return self._postprocess_data(x, t, y_f, y_cf, dataset, standardize)

    def load_treatment_and_outcome(self, covariates, file, standardize=True):
        out = pd.read_csv(file, header=0, sep=',')
        dataset = covariates.join(out, how='inner').drop(columns=['mu0', 'mu1'])
        t = dataset['z'].values
        y_f = np.where(t == 1, dataset['y1'].values, dataset['y0'].values)
        y_cf = np.where(t == 1, dataset['y0'].values, dataset['y1'].values)
        x = dataset.values[:, :-3]
        t, y_f, y_cf, x = t.reshape(-1, 1).astype(float), y_f.reshape(-1, 1), y_cf.reshape(-1, 1), x
        return self._postprocess_data(x, t, y_f, y_cf, dataset, standardize)

    def get_data(self):
        if self.year == 2018:
            x_raw = pd.read_csv(self.cov_data_path, index_col='sample_id', header=0, sep=',')
        elif self.year == 2016:
            x_raw = pd.read_csv(self.cov_data_path, header=0, sep=',')
            x_raw = pd.get_dummies(x_raw)

        datasets = []
        if self.year == 2018:

            for file_f, file_cf in zip(self.simulation_files_f, self.simulation_files_cf):
                datasets.append(self.load_treatment_and_outcome_f_cf(x_raw, file_f, file_cf))
        elif self.year == 2016:
            for file in self.simulation_files:
                datasets.append(self.load_treatment_and_outcome(x_raw, file))
        return datasets
