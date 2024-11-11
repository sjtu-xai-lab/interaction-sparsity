"""
1. compute the derived upper bound of the number of valid interactions（up to M-th order） (Table 2)
2. compute the real number of valid interactions （up to M-th order） (Table 2)
3. compute average value of p (Fig 5(b))
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
from demo.prove_symbolic_utils import (is_mono_increasing, compute_p_ref_order,
                                       compute_statistics_all_orders, compute_delta_bound,
                                       compute_real_R_k, DeltakEmptyException, DeltakTooSmallException,
                                       LambdaCloseToZeroException, compute_mean_vS)

parser = argparse.ArgumentParser()
parser.add_argument('--ref_order', type=int, default=1)
parser.add_argument('--model', type=str, default="llama", choices=["opt","llama","aquila"])
args = parser.parse_args()


FONT = 15

use_stop_words = True
compute_bound = True

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


n = 10
orders = np.arange(0, n+1)


if __name__ == '__main__':
    M = 9
    all_p = []
    all_real_num_valid_concepts = []
    all_bounds = []
    all_p_plus_delta = []

    for name in sorted(os.listdir(root)):

        with open(os.path.join(root, name, "infernece.txt"), 'r') as f:
            last_line = f.readlines()[-2]
            inference_word = last_line.strip().split()[-1]
        if use_stop_words and inference_word in stop_words:
            continue

        # print(f"========{name}=======")
        vS = np.load(os.path.join(root, name, "rewards.npy"))
        uS = vS - vS[0]
        mean_uS_this_sample = compute_mean_vS(uS, num_players=n)

        increasing = is_mono_increasing(mean_uS_this_sample)
        if increasing: # we only consider samples that satisfy the monotonicity assumption
            p_this_sample = compute_p_ref_order(mean_uS_this_sample, num_players=n, ref_order=args.ref_order)
            all_p.append(p_this_sample)

            if uS[-1] > vN_thres: # we only consider samples with v(N)-v(empty)>threshold
                if compute_bound:
                    p_floor = int(np.floor(p_this_sample))
                    all_IS = np.load(os.path.join(root, name, "Iand.npy"))
                    TAU = 0.05 * np.abs(all_IS).max()

                    try:
                        A_k_dict, \
                        A_k_sign_dict, \
                        eta_k_dict, \
                        q_k_dict, \
                        all_a_lt_p_floor_dict, \
                        delta_k_dict, \
                        lambda_k_dict, \
                        bar_u1 = compute_statistics_all_orders(all_IS, all_uS=uS, num_players=n, p=p_this_sample)
                    except DeltakTooSmallException as e:
                        # print("caught exception: ", e)
                        continue
                    except DeltakEmptyException as e:
                        # print("caught exception: ", e)
                        continue

                    try:
                        delta_bound_best, m0_best = compute_delta_bound(lambda_k_dict, all_a_lt_p_floor_dict, p=p_this_sample,
                                                                        p_floor=p_floor, num_players=n, M=M)
                    except LambdaCloseToZeroException as e:
                        # print("caught exception: ", e)
                        continue
                    except ValueError as e:
                        # print("caught value error: ", e)
                        continue

                    real_R_k_dict = compute_real_R_k(all_IS, num_players=n, tau=TAU)
                    num_valid_concepts = 0
                    for k in range(1, M + 1):
                        num_valid_concepts += real_R_k_dict[k]
                    all_real_num_valid_concepts.append(num_valid_concepts)

                    num_valid_concepts_predict = 0
                    for k in range(1, M + 1): # when computing the bound, we only consider orders up to M, because we have assumed interactions higher than M-th order are all zero
                        sum = lambda_k_dict[k] * (n ** (p_this_sample + delta_bound_best))

                        for i in range(0, p_floor):
                            sum += all_a_lt_p_floor_dict[k][i] * (n ** i)
                        R_k_bound = bar_u1 / (TAU * abs(eta_k_dict[k])) * abs(sum)
                        num_valid_concepts_predict += R_k_bound

                    all_p_plus_delta.append(p_this_sample + delta_bound_best)
                    all_bounds.append(num_valid_concepts_predict)


    print(f"real number of valid concepts mean={np.mean(all_real_num_valid_concepts)}, std={np.std(all_real_num_valid_concepts)}")
    print(f"bound of valid concepts mean={np.mean(all_bounds)}, std={np.std(all_bounds)}")
    print(f"bound-real mean={np.mean(np.array(all_bounds) - np.array(all_real_num_valid_concepts))}, "
          f"std={np.std(np.array(all_bounds) - np.array(all_real_num_valid_concepts))}")
    print(f"p+delta mean={np.mean(all_p_plus_delta)}, std={np.std(all_p_plus_delta)}")

    all_p = np.array(all_p)
    print("\n" + "="*20 + " statistics of the value of p " + "="*20)
    print("all_p.mean()", all_p.mean())
    print("all_p.std()", all_p.std())


