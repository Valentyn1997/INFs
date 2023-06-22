import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import ShuffleSplit, KFold
from scipy.stats import wasserstein_distance

from src.models.utils import subset_by_indices
from src.visualization.utils import plot_interventional_densities


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of train_data_dict
    seed_everything(args.exp.seed)
    dataset = instantiate(args.dataset, _recursive_=True)
    data_dicts = dataset.get_data() if args.dataset.collection else [dataset.get_data()]
    if args.dataset.dataset_ix is not None:
        data_dicts = [data_dicts[args.dataset.dataset_ix]]
        specific_ix = True
    else:
        specific_ix = False

    for ix, data_dict in enumerate(data_dicts):

        # Train-test split
        if hasattr(args.dataset, 'k_fold'):
            rs = KFold(n_splits=args.dataset.k_fold, random_state=args.exp.seed, shuffle=True)
        else:
            rs = ShuffleSplit(n_splits=args.dataset.n_shuffle_splits, random_state=args.exp.seed,
                              test_size=args.dataset.test_size)
        for split_ix, (train_index, test_index) in enumerate(rs.split(data_dict['cov_f'])):
            results = {}

            train_data_dict, test_data_dict = subset_by_indices(data_dict, train_index), subset_by_indices(data_dict, test_index)

            # Initialisation of model
            args.dataset.dataset_ix = ix if not specific_ix else args.dataset.dataset_ix
            args.dataset.split_ix = split_ix
            args.model.nuisance_bound = float(train_data_dict['out_f'].max() - train_data_dict['out_f'].min()) + 5.0
            args.model.target_bound = float(train_data_dict['out_f'].max() - train_data_dict['out_f'].min()) + 5.0
            model = instantiate(args.model, args, _recursive_=True)

            # Finetuning for the first split
            if args.model.tune_hparams and split_ix == 0:
                model.finetune(train_data_dict, {'cpu': 1.0, 'gpu': 0.0})

            # Fitting the model
            model.fit(train_data_dict=train_data_dict, log=args.exp.logging)

            # Evaluation
            if 'out_pot_0' in train_data_dict and 'out_pot_1' in train_data_dict:
                for a in args.model.treat_options:
                    # In-sample potential log-prob
                    treat_pot = float(a) * np.ones((train_data_dict[f'out_pot_{a}'].shape[0], ))
                    try:
                        in_sample_log_prob = model.inter_log_prob(treat_pot, train_data_dict[f'out_pot_{a}']).mean()
                    except Exception:
                        in_sample_log_prob = np.nan
                    logger.info(f"In-sample {args.model.name} log_prob [{a}]: {in_sample_log_prob}")
                    results.update({f'in_sample_log_prob_{a}': in_sample_log_prob})

                    # In-sample empirical Wasserstein distance
                    # if args.dataset.name == 'ihdp':
                    #     in_pot_sample = model.inter_sample(treat_pot, 1)
                    #     in_wd = wasserstein_distance(in_pot_sample.squeeze(), train_data_dict[f'out_pot_{a}'].squeeze())
                    #     results.update({f'in_sample_wd_{a}': in_wd})

                    # Out-sample potential log-prob
                    treat_pot = float(a) * np.ones((test_data_dict[f'out_pot_{a}'].shape[0],))
                    try:
                        out_sample_log_prob = model.inter_log_prob(treat_pot, test_data_dict[f'out_pot_{a}']).mean()
                    except Exception:
                        out_sample_log_prob = np.nan
                    logger.info(f"Out-sample {args.model.name} log_prob [{a}]: {out_sample_log_prob}")
                    results.update({f'out_sample_log_prob_{a}': out_sample_log_prob})

                    # Out-sample empirical Wasserstein distance
                    # if args.dataset.name == 'ihdp':
                    #     out_pot_sample = model.inter_sample(treat_pot, 1)
                    #     out_wd = wasserstein_distance(out_pot_sample.squeeze(), test_data_dict[f'out_pot_{a}'].squeeze())
                    #     results.update({f'out_sample_wd_{a}': out_wd})

                    # Whole-sample ATE estimation
                    try:
                        eps_ate = np.abs(data_dict[f'out_pot_{a}'].mean(0) - model.inter_mean(float(a)))
                    except Exception:
                        eps_ate = np.nan
                    logger.info(f"{args.model.name} eps-ATE [{a}]: {eps_ate}")
                    if args.model.dim_out == 1:
                        results.update({f'eps_ate_{a}': eps_ate})

                model.mlflow_logger.log_metrics(results, step=split_ix) if args.exp.logging else None
            model.mlflow_logger.experiment.set_terminated(model.mlflow_logger.run_id) if args.exp.logging else None

            # try:
            # filename = dataset.simulation_files_f[args.dataset.dataset_ix].split('/')[-1].split('.')[0].split('_')[-1]
            # filename = dataset.simulation_files[args.dataset.dataset_ix].split('/')[-1].split('.')[0].split('_')[-1]
            # filename = 'hcmnist-results' #f'{filename}_{split_ix}'
            plot_interventional_densities(data_dict, model, out_scaler=dataset.out_scaler, save_to_dir=True, filename=filename) \
                if args.exp.plotting else None
            # except Exception:
            #     pass

    return results


if __name__ == "__main__":
    main()
