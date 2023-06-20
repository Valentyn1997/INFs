import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from src import ROOT_PATH

import pandas as pd

# rc('font', **{'family': 'libertine'})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsfonts}')


def plot_interventional_densities(data_dict, model, save_to_dir=False, out_scaler=None, filename=None):
    if data_dict['out_f'].shape[1] == 1:  # 1D outcome
        if 'out_pot_0' in data_dict and 'out_pot_1' in data_dict:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(9, 3))

            ax[0].grid()
            ax[1].grid()

            ax[0].hist(data_dict['out_pot_0'], alpha=0.4, density=True, bins=30, label='$\\mathbb{P}(Y[0] = y)$',
                       color='tab:blue')
            ax[1].hist(data_dict['out_pot_1'], alpha=0.4, density=True, bins=30, label='$\\mathbb{P}(Y[1] = y)$',
                       color='tab:orange')

            sns.distplot(data_dict['out_f'][data_dict['treat_f'] == 0.0], hist=False,
                         label='$\\mathbb{P}(Y = y \\mid A = 0)$', ax=ax[0], color='tab:blue')
            sns.distplot(data_dict['out_f'][data_dict['treat_f'] == 1.0], hist=False, label='$\\mathbb{P}(Y = y \\mid A = 1)$',
                         ax=ax[1], color='tab:orange')

            x = np.linspace(data_dict['out_f'].min(), data_dict['out_f'].max(), 500)
            ax[0].plot(x, np.exp(model.inter_log_prob(np.zeros((500,)), x)),
                       label='$\\hat{\\mathbb{P}}^{\\mathrm{INFs}}(Y[0] = y)$', color='tab:blue')
            ax[1].plot(x, np.exp(model.inter_log_prob(np.ones((500,)), x)),
                       label='$\\hat{\\mathbb{P}}^{\\mathrm{INFs}}(Y[1] = y)$', color='tab:orange')

            ax[0].lines[0].set_linestyle("--")
            ax[1].lines[0].set_linestyle("--")

            ax[0].set_xlabel('$y$')
            ax[1].set_xlabel('$y$')

            ax[0].set_ylabel(None)
            ax[1].set_ylabel(None)

            ax[0].legend()
            ax[1].legend()

            ax[0].set_ylim(0.0, 0.7)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True, figsize=(4.5, 3))

            ax.grid()
            sns.distplot(out_scaler.inverse_transform(data_dict['out_f'][data_dict['treat_f'] == 0.0].reshape(-1, 1)), hist=False,
                         label='$\\mathbb{P}(Y = y \\mid A = 0)$', ax=ax, color='tab:blue')
            sns.distplot(out_scaler.inverse_transform(data_dict['out_f'][data_dict['treat_f'] == 1.0].reshape(-1, 1)), hist=False,
                         label='$\\mathbb{P}(Y = y \\mid A = 1)$', ax=ax, color='tab:orange')

            x = np.linspace(data_dict['out_f'].min(), data_dict['out_f'].max(), 500)
            x_unscaled = out_scaler.inverse_transform(x.reshape(-1, 1))
            ax.plot(x_unscaled,
                    np.exp(model.inter_log_prob(np.zeros((500,)), x) - np.log(out_scaler.scale_).sum()),
                    label='$\\hat{\\mathbb{P}}^{\\mathrm{INFs}}(Y[0] = y)$', color='tab:blue')
            ax.plot(x_unscaled,
                    np.exp(model.inter_log_prob(np.ones((500,)), x) - np.log(out_scaler.scale_).sum()),
                    label='$\\hat{\\mathbb{P}}^{\\mathrm{INFs}}(Y[1] = y)$', color='tab:orange')

            ax.lines[0].set_linestyle("--")
            ax.lines[1].set_linestyle("--")
            ax.set_title('$n_{\\mathrm{knots,T}} = 10$')
            ax.set_xlabel('$y$, per-capita cigarette sales from 1970 to 2000 (in packs)')

            ax.set_ylabel(None)

            ax.legend()

            # ax.set_ylim(0.0, 0.7)

        fig.tight_layout()
        if save_to_dir:
            plt.savefig(f'{ROOT_PATH}/reports/{filename}.pdf', bbox_inches='tight', pad_inches=0)
    else:  # 2D outcome
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(9, 3))
        grid_size = 20
        x = np.linspace(data_dict['out_f'][:, 0].min(), data_dict['out_f'][:, 0].max(), grid_size)
        y = np.linspace(data_dict['out_f'][:, 1].min(), data_dict['out_f'][:, 1].max(), grid_size)

        xx, yy = np.meshgrid(x, y)
        xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

        ax[0].contour(xx, yy, np.exp(model.inter_log_prob(np.zeros((grid_size * grid_size,)), xy)).reshape(grid_size, grid_size))
        sns.kdeplot(pd.DataFrame(data_dict['out_pot_0']), x=0, y=1, label=0, color='tab:blue', ax=ax[0])
        ax[0].legend()

        ax[1].contour(xx, yy, np.exp(model.inter_log_prob(np.ones((grid_size * grid_size,)), xy)).reshape(grid_size, grid_size))
        sns.kdeplot(pd.DataFrame(data_dict['out_pot_1']), x=0, y=1, label=1, color='tab:orange', ax=ax[1])
        ax[1].legend()

    plt.show()
