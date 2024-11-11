"""
Compute real number of valid concepts of each order for each sample (count all orders, not only the first M orders)
(Table 1, second and third row)
"""
import os
Project_root = os.getcwd()
import sys
sys.path.append(Project_root)

import numpy as np
import argparse
import matplotlib

matplotlib.use('agg')
from demo.stop_words import stop_words
from demo.prove_symbolic_utils import (is_mono_increasing, compute_real_R_k, compute_mean_vS)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="aquila", choices=["opt","llama","aquila"])
args = parser.parse_args()

if args.model == "opt":
    root = "results/interaction_opt/gt-log-odds-v0_mode=q_qthres=0.04_lr=1e-05_epoch=20000"
    vN_thres = 1.0
elif args.model == "llama":
    root = "results/interaction_llama/gt-log-odds-v0_mode=q_qthres=0.04_lr=1e-05_epoch=20000"
    vN_thres = 2.0
elif args.model == "aquila":
    root = "results/interaction_aquila/gt-log-odds-v0_mode=q_qthres=0.04_lr=1e-05_epoch=20000"
    vN_thres = 3.0
else:
    raise NotImplementedError(f"model [{args.model}] is not implemented")


FONT = 15

n = 10
orders = np.arange(0, n+1)
use_stop_words = True


if __name__ == '__main__':
    num_tot = 0
    num_mono_increasing = 0
    num_salient_concepts_mono_samples = []
    num_salient_concepts_nonmono_samples = []

    for name in sorted(os.listdir(root)):

        with open(os.path.join(root, name, "infernece.txt"), 'r') as f:
            last_line = f.readlines()[-2]
            inference_word = last_line.strip().split()[-1]

        if use_stop_words and inference_word in stop_words:
            continue

        num_tot += 1

        # print(f"========{name}=======")
        vS = np.load(os.path.join(root, name, "rewards.npy"))
        uS = vS - vS[0] # u(S) = v(S) - v(empty)
        mean_uS_this_sample = compute_mean_vS(uS, num_players=n) # compute the mean value of u(S) for each order

        all_IS = np.load(os.path.join(root, name, "Iand.npy"))

        TAU = 0.05 * np.abs(all_IS).max() # threshold tau for salient concepts

        real_R_k_dict = compute_real_R_k(all_IS, num_players=n, tau=TAU) # a dict containing the value of R^(k) of each order

        num_valid_concepts = 0
        for k in range(1, n + 1): # count valid concepts of all orders (except order 0, which is meaningless)
            num_valid_concepts += real_R_k_dict[k]

        increasing = is_mono_increasing(mean_uS_this_sample)
        if increasing:
            num_mono_increasing += 1
            if uS[-1] > vN_thres:
                num_salient_concepts_mono_samples.append(num_valid_concepts)
        else:
            if uS[-1] > vN_thres:
                num_salient_concepts_nonmono_samples.append(num_valid_concepts)

    num_salient_concepts_mono_samples = np.array(num_salient_concepts_mono_samples)
    num_salient_concepts_nonmono_samples = np.array(num_salient_concepts_nonmono_samples)

    print("-"*30)

    if len(num_salient_concepts_mono_samples) > 0:
        print(f"num_salient_concepts on monotonic samples mean={num_salient_concepts_mono_samples.mean()} "
              f"std={num_salient_concepts_mono_samples.std()}")
    if len(num_salient_concepts_nonmono_samples) > 0:
        print(f"num_salient_concepts on non-monotonic samples mean={num_salient_concepts_nonmono_samples.mean()} "
          f"std={num_salient_concepts_nonmono_samples.std()}")




