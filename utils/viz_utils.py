import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import numpy as np
import seaborn as sns

import torch
from torchvision.transforms import ToTensor, Resize, ToPILImage

from utils.turbo_cmap import turbo_cm


def get_example_figure(ensemble_data, endd_data, hist_data):
    fig = plt.figure(figsize=(17.77, 10))
    plt.rc('font', size=30)
    gs = gridspec.GridSpec(
        ncols=4, nrows=3,
        figure=fig, width_ratios=[1.0] * 4
    )
    for i in range(3):
        for k in range(4):
            ax = fig.add_subplot(gs[i, k])
            if i == 0:
                title_text = {
                    0: "Input", 1: "Error", 2: "Total", 3: "Knowledge"
                }
                ax.set_title(title_text[k])
            if k == 0:
                axis_text = {
                    0: "ENSM", 1: "EnD$^2$", 2: "Statistics"
                }
                ax.set_ylabel(axis_text[i], size='large')
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            if i == 0:
                cshow_data = ensemble_data[k]
            elif i == 1:
                cshow_data = endd_data[k]
            else:
                cshow_data = hist_data[k]
            plt.imshow(
                ToPILImage()(cshow_data.squeeze().float()),
                aspect='auto'
            )
    plt.subplots_adjust(
        hspace=0.025, wspace=0.025,
        top=0.93, bottom=0.025,
        left=0.05, right=(1.0 - 0.025)
    )
    return plt.gcf()


def get_tensor_with_histograms(results_dict, plot_names, model_names):
    sns.set_style("white")
    colors = [sns.color_palette()[i] for i in range(3)]

    all_hists = []
    c_xlim = None
    for ind in range(len(results_dict[model_names[1]][0])):
        # Add empty black image
        all_hists.append(
            torch.zeros_like(results_dict[model_names[1]][0][ind])
        )
        for key in results_dict[model_names[1]][1].keys():
            # Add seaborn histogram for every image
            plt.figure(figsize=(13.33, 10))
            plt.rc('font', size=50)

            max_p = 95 if key == 'expected_pairwise_kl' else 100
            c_max = np.percentile(
                np.concatenate([
                    results_dict[m_name][1][key][ind]
                    for m_name in set(model_names) - set(['gaussian'])
                ]), max_p
            )
            for i, (model, m_name) in enumerate(zip(
                model_names, plot_names
            )):
                if model == 'gaussian':
                    continue
                c_show = results_dict[model][1][key][ind]
                c_show[c_show > c_max] = c_max
                sns.distplot(
                    c_show, label=m_name,
                    kde_kws={"lw": 3}, color=colors[i],
                    norm_hist=True
                )
            plt.grid()
            plt.legend()
            if key == 'total_variance':
                plt.xlim((c_xlim[0], c_xlim[1]))
            plt.savefig('plots/tmp.png', tight_layout=True, dpi=400)
            c_xlim = plt.gca().get_xlim()
            plt.clf()

            im = Image.open('plots/tmp.png').convert("RGB")
            res = Resize((
                results_dict[model_names[1]][0][ind].size(1),
                results_dict[model_names[1]][0][ind].size(2))
            )(im)
            res = ToTensor()(res)
            all_hists.append(res)

    return torch.cat([vec.unsqueeze(0) for vec in all_hists])


def colorize(value, vmin=0.0, vmax=1.0, cmap='turbo', numpy=False):
    if not numpy:
        value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    value = np.clip(value, 0.0, 1.0)

    if cmap != 'turbo':
        cmapper = matplotlib.cm.get_cmap(cmap)
    else:
        cmapper = turbo_cm
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))
