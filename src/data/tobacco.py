import pandas as pd
import numpy as np
import pyreadr
import logging

from sklearn import preprocessing

from src import ROOT_PATH

START_TIME = 1970
INTERVENTION_TIME = 1989
STOP_TIME = 2001

logger = logging.getLogger(__name__)


class TobaccoControlProgram:

    def __init__(self, normalize_out=False, **kwargs):
        df_outcome_raw = pd.read_csv(f'{ROOT_PATH}/data/tobacco_control/prop99.csv')
        df_outcome_raw = df_outcome_raw[df_outcome_raw['SubMeasureDesc'] == 'Cigarette Consumption (Pack Sales Per Capita)']
        self.df_outcome = pd.DataFrame(df_outcome_raw.pivot_table(values='Data_Value', index='LocationDesc',
                                                                  columns=['Year']).to_records())

        rda_predictors = pyreadr.read_r(f'{ROOT_PATH}/data/tobacco_control/smoking.rda')
        self.df_predictors = pd.DataFrame(list(rda_predictors.values())[0])

        logger.info(f'In the original dataset there are {self.df_outcome.LocationDesc.unique().shape[0]} states')
        # Section 3.2 in the paper
        bad_states = ['Massachusetts', 'Arizona', 'Oregon', 'Florida', 'Alaska', 'Hawaii', 'Maryland',
                      'Michigan', 'New Jersey', 'New York', 'Washington', 'District of Columbia']

        self.df_outcome.drop(self.df_outcome[self.df_outcome['LocationDesc'].isin(bad_states)].index, inplace=True)
        # ca_id = self.df_outcome[self.df_outcome['LocationDesc'] == 'California'].index.item()
        self.df_outcome = self.df_outcome.reset_index()
        self.df_outcome = self.df_outcome.rename(columns={'index': 'org_index'})
        logger.info(f'After filtering out some states, '
                    f'we are left with {self.df_outcome.LocationDesc.unique().shape[0]} states (including California):')

        df_outcome_ca = self.df_outcome.loc[self.df_outcome['LocationDesc'] == 'California', :]
        df_outcome_control = self.df_outcome.loc[self.df_outcome['LocationDesc'] != 'California', :]

        ca_outcomes_pre = df_outcome_ca.loc[:, [str(i) for i in list(range(START_TIME, INTERVENTION_TIME))]].values.reshape(-1, 1)
        control_outcomes_pre = \
            df_outcome_control.loc[:, [str(i) for i in list(range(START_TIME, INTERVENTION_TIME))]].values.transpose()

        ca_outcomes_post = df_outcome_ca.loc[:, [str(i) for i in list(range(INTERVENTION_TIME, STOP_TIME))]].values.reshape(-1, 1)
        control_outcomes_post = \
            df_outcome_control.loc[:, [str(i) for i in list(range(INTERVENTION_TIME, STOP_TIME))]].values.transpose()

        self.Z0 = control_outcomes_pre
        self.Z1 = ca_outcomes_pre
        self.Y0 = control_outcomes_post
        self.Y1 = ca_outcomes_post

        control_predictors = []
        for state in self.df_outcome['LocationDesc'].unique():
            state_predictor_vec = self.extract_predictor_vec(state)
            if state == 'California':
                ca_predictors = state_predictor_vec
            else:
                control_predictors += [state_predictor_vec]

        control_predictors = np.stack(control_predictors, axis=1)

        self.X0 = control_predictors
        self.X1 = ca_predictors

        self.normalize_out = normalize_out
        self.out_scaler = preprocessing.StandardScaler()

    def extract_predictor_vec(self, state):
        state_id_predictors_df = self.df_outcome[self.df_outcome['LocationDesc'] == state].index.item() + 1
        df_predictors_state = self.df_predictors[self.df_predictors['state'] == state_id_predictors_df]
        df_predictors_state = df_predictors_state.fillna(method='ffill').fillna(method='bfill')
        beer_predictor = df_predictors_state.loc[
            (df_predictors_state['year'] >= START_TIME) & (df_predictors_state['year'] < STOP_TIME), 'beer']
        age15to24_predictor = df_predictors_state.loc[
            (df_predictors_state['year'] >= START_TIME) & (df_predictors_state['year'] < STOP_TIME), 'age15to24']
        retprice_predictor = df_predictors_state.loc[
            (df_predictors_state['year'] >= START_TIME) & (df_predictors_state['year'] < STOP_TIME), 'retprice']
        lnincome_predictor = df_predictors_state.loc[
            (df_predictors_state['year'] >= START_TIME) & (df_predictors_state['year'] < STOP_TIME), 'lnincome']

        return np.array([lnincome_predictor, age15to24_predictor, retprice_predictor, beer_predictor]).T

    def get_data(self):
        out_0_f = np.expand_dims(np.concatenate([self.Z0, self.Y0], axis=0), -1)
        cov_0_f = self.X0
        time_f = np.repeat(np.expand_dims(np.arange(0, 31, dtype=float), (1, 2)), 38, 1)
        cov_0_f = np.concatenate([cov_0_f, time_f], axis=2)
        out_0_f = out_0_f.reshape(-1, 1)
        cov_0_f = cov_0_f.reshape(-1, 5)

        out_0_cal_f = self.Z1
        out_1_cal_f = self.Y1
        cov_cal_f = self.X1
        time_cal_f = np.expand_dims(np.arange(0, 31, dtype=float), 1)
        cov_cal_f = np.concatenate([cov_cal_f, time_cal_f], axis=1)

        out_f = np.concatenate([out_0_f, out_0_cal_f, out_1_cal_f], axis=0)
        cov_f = np.concatenate([cov_0_f, cov_cal_f], axis=0)
        treat_f = np.concatenate([np.zeros((len(out_0_f) + len(out_0_cal_f), 1)), np.ones((len(out_1_cal_f), 1))], 0)

        if self.normalize_out:
            out_f = self.out_scaler.fit_transform(out_f.reshape(-1, 1))

        return {
            'cov_f': cov_f,
            'treat_f': treat_f,
            'out_f': out_f,
        }
