from hydra.experimental import initialize, compose
import logging

from runnables.train import main

logging.basicConfig(level='info')


class TestKDEPolynomialNormal:
    def test_kde(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=polynomial_normal",
                                                                 "+model=kde",
                                                                 "+model/polynomial_normal_hparams/kde='5.0'",
                                                                 "dataset.k_fold=2", "dataset.n_samples=200",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.plotting=False",
                                                                 "model.num_epochs=5"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2