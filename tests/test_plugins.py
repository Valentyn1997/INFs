from hydra.experimental import initialize, compose
import logging

from runnables.train import main

logging.basicConfig(level='info')


class TestPluginsPolynomialNormal:
    def test_gauss_tarnet_plugin(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=polynomial_normal",
                                                                 "+model=gauss_tarnet_plugin",
                                                                 "dataset.k_fold=2", "dataset.n_samples=200",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.plotting=False",
                                                                 "model.num_epochs=5"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2

    def test_mdn_plugin(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=polynomial_normal",
                                                                 "+model=mdn_plugin",
                                                                 "+model/polynomial_normal_hparams/mdn_plugin='5.0'",
                                                                 "dataset.k_fold=2", "dataset.n_samples=200",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.plotting=False",
                                                                 "model.num_epochs=5"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2
