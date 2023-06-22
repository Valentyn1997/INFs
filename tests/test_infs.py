from hydra.experimental import initialize, compose
import logging

from runnables.train import main

logging.basicConfig(level='info')


class TestINFsPolynomialNormal:
    def test_infs_aiptw(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=polynomial_normal",
                                                                 "+model=infs_aiptw",
                                                                 "+model/polynomial_normal_hparams/infs='2.5'",
                                                                 "dataset.k_fold=2", "dataset.n_samples=200",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.plotting=False",
                                                                 "model.num_epochs=5",
                                                                 "model.target_num_epochs=5"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2

    def test_infs_covariate_adjusted(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=polynomial_normal",
                                                                 "+model=infs_covariate_adjusted",
                                                                 "+model/polynomial_normal_hparams/infs='2.5'",
                                                                 "dataset.k_fold=2", "dataset.n_samples=200",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.plotting=False",
                                                                 "model.num_epochs=5",
                                                                 "model.target_num_epochs=5"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2

    def test_infs_plugin(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=polynomial_normal",
                                                                 "+model=infs_plugin",
                                                                 "+model/polynomial_normal_hparams/infs='2.5'",
                                                                 "dataset.k_fold=2", "dataset.n_samples=200",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.plotting=False",
                                                                 "model.num_epochs=5"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2


class TestCNFTruncatedSeries:
    def test_cnf_truncated_series_aiptw(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=polynomial_normal",
                                                                 "+model=cnf_truncated_series_aiptw",
                                                                 "+model/polynomial_normal_hparams/infs='2.5'",
                                                                 "dataset.k_fold=2", "dataset.n_samples=200",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.plotting=False",
                                                                 "model.num_epochs=5"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2
