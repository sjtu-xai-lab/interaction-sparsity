"""
Compute how many samples satisfy the monotonicity assumption (Table 1 first row)
"""
import os
Project_root = os.getcwd()
import sys
sys.path.append(Project_root)

import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
from demo.prove_symbolic_utils import compute_mean_vS, mkdir, is_mono_increasing
from demo.stop_words import stop_words

FONT = 15

use_stop_words = True

# change model here
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

n = 10
orders = np.arange(0, n+1)


if __name__ == '__main__':
    not_increasing_list = []
    count_valid = 0
    count_increasing = 0
    for name in sorted(os.listdir(root)):
        with open(os.path.join(root, name, "infernece.txt"), 'r') as f:
            last_line = f.readlines()[-2]
            inference_word = last_line.strip().split()[-1]

        if use_stop_words and inference_word in stop_words: # filter out stop words
            continue

        vS = np.load(os.path.join(root, name, "rewards.npy"))
        uS = vS - vS[0] # u(S) = v(S) - v(empty)

        count_valid += 1

        mean_uS_order = compute_mean_vS(uS, num_players=n) # compute the mean value of u(S) for each order

        increasing = is_mono_increasing(mean_uS_order)
        if increasing:
            count_increasing += 1
        else:
            not_increasing_list.append(f"sample: {name}, inference words: {inference_word}\n")
            # print(f"sample {name} not increasing")

    print(f"count valid={count_valid}, count increasing={count_increasing}, ratio={count_increasing/count_valid}")

