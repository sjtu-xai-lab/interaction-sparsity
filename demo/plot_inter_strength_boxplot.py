"""
Use box and whisker plot to show the approximate distribution of |I(S)| of each order (Fig 4)
"""
import os
Project_root = os.getcwd()
import sys
sys.path.append(Project_root)

import numpy as np
import argparse
import matplotlib
import math
matplotlib.use('agg')
import matplotlib.pyplot as plt
from nouse.plot_for_single import compute_mean_vS
from demo.stop_words import stop_words
from demo.prove_symbolic_utils import (mkdir, is_mono_increasing)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="aquila", choices=["opt","llama","aquila"])
args = parser.parse_args()


FONT = 35
show_xy_labels = True
use_stop_words = True

if args.model == "opt":
    root = "results/interaction_opt/gt-log-odds-v0_mode=pq_qthres=0.04_lr=1e-06_epoch=50000"
    vN_thres = 1.0
    sample_indices = [1, 33]
elif args.model == "llama":
    root = "results/interaction_llama/gt-log-odds-v0_mode=pq_qthres=0.04_lr=1e-06_epoch=50000"
    vN_thres = 2.0
    sample_indices = [3, 19]
elif args.model == "aquila":
    root = "results/interaction_aquila/gt-log-odds-v0_mode=pq_qthres=0.04_lr=1e-06_epoch=50000"
    vN_thres = 3.0
    sample_indices = [3, 18]
else:
    raise NotImplementedError


n = 10
orders = np.arange(0, n+1)

save_path = os.path.join(root, f"../inter_strength_boxplot")
mkdir(save_path)


if __name__ == '__main__':
    x = np.arange(1, n+1)
    xticks = [1, n]

    for i, sample_idx in enumerate(sample_indices):
        name = f"sample{sample_idx}"

        with open(os.path.join(root, name, "infernece.txt"), 'r') as f:
            last_line = f.readlines()[-2]
            inference_word = last_line.strip().split()[-1]

        if use_stop_words and inference_word in stop_words:
            print(f"*** skip {name} ***")
            continue

        vS = np.load(os.path.join(root, name, "rewards.npy"))
        uS = vS - vS[0]  # u(S) = v(S) - v(empty)
        mean_uS_this_sample = compute_mean_vS(uS, num_players=n)  # compute the mean value of u(S) for each order

        all_IS = np.load(os.path.join(root, name, "Iand.npy"))

        increasing = is_mono_increasing(mean_uS_this_sample)
        if increasing: # only look at samples that satisfy the monotonicity assumption
            if uS[-1] > vN_thres: # only look at samples with v(N)-v(empty)>threshold
                inter_strength_by_order = []
                count = 0
                for order in range(0, n + 1):
                    num = math.comb(n, order)
                    all_IS_this_order = all_IS[count: count + num]
                    inter_strength_by_order.append(np.abs(all_IS_this_order))
                    count = count + num

                plt.figure(figsize=(8,5))
                # we do not consider order 0
                bplot = plt.boxplot(inter_strength_by_order[1:],
                                    patch_artist=True,
                                    medianprops=dict(color='black'),
                                    showfliers=False)

                for patch in bplot['boxes']:
                    patch.set_facecolor("#436EEE")

                if show_xy_labels:
                    plt.xlabel(r"order $m$", fontsize=FONT)
                    plt.ylabel(r"$I_{str}^{(m)}$", fontsize=FONT)
                plt.xticks(xticks, xticks)
                plt.tick_params(labelsize=FONT)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"example{i}.png"))
                plt.savefig(os.path.join(save_path, f"example{i}.svg"))
                plt.close()
