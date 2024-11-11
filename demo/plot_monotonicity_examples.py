"""
Choose a few monotonic samples as examples to show the curve of \bar{u}^(m) with m (Fig 5(a))
"""
import os
Project_root = os.getcwd()
import sys
sys.path.append(Project_root)

import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
from demo.prove_symbolic_utils import (compute_mean_vS, mkdir, is_mono_increasing,
                                       plot_multiple_curves, compute_std_vS)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="aquila", choices=["opt","llama","aquila"])
args = parser.parse_args()


if args.model == "opt":
    root = "results/interaction_opt/gt-log-odds-v0_mode=q_qthres=0.04_lr=1e-05_epoch=20000"
elif args.model == "llama":
    root = "results/interaction_llama/gt-log-odds-v0_mode=q_qthres=0.04_lr=1e-05_epoch=20000"
elif args.model == "aquila":
    root = "results/interaction_aquila/gt-log-odds-v0_mode=q_qthres=0.04_lr=1e-05_epoch=20000"
else:
    raise NotImplementedError(f"model [{args.model}] is not implemented")


save_plot_folder = "mean_uS_plot_examples_with_std"
save_path = os.path.join(root, "..", save_plot_folder)
mkdir(save_path)

n = 10
orders = np.arange(0, n+1)

num_samples_selected = 5
seed = 1


if __name__ == '__main__':
    uS_examples_list = []
    std_examples_list = []
    name_list = []

    for name in sorted(os.listdir(root)):
        vS = np.load(os.path.join(root, name, "rewards.npy"))
        uS = vS - vS[0]
        mean_uS_order = compute_mean_vS(uS, num_players=n)
        std_uS_order = compute_std_vS(uS, num_players=n)

        # we only look at samples that satisfy the monotonicity assumption
        if is_mono_increasing(mean_uS_order):
            uS_examples_list.append(mean_uS_order)
            std_examples_list.append(std_uS_order)
            name_list.append(name)

    uS_examples_list = np.stack(uS_examples_list, axis=0)
    std_examples_list = np.stack(std_examples_list, axis=0)

    np.random.seed(seed)
    select_index = np.random.choice(np.arange(len(uS_examples_list)), size=num_samples_selected) # randomly select 5 samples

    plot_multiple_curves(orders, uS_examples_list[select_index],
                         save_root=save_path, save_name="mono_examples",
                         xlabel=r"order $m$", ylabel=r"$\bar{u}^{(m)}$",
                         linewidth=5, font=30, xticks=[0, 10],
                         std_bars=std_examples_list[select_index])

