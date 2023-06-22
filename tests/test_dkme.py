from hydra.experimental import initialize, compose
import logging

from runnables.train import main

logging.basicConfig(level='info')


class TestDKMEPolynomialNormal:
    def test_dkme(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=polynomial_normal",
                                                                 "+model=dkme",
                                                                 "+model/polynomial_normal_hparams/dkme='2.5'",
                                                                 "dataset.k_fold=2", "dataset.n_samples=200",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.plotting=False"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2