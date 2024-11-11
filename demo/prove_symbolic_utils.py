import math
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from typing import Union, List

FONT = 20

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_mono_increasing(arr):
    flag = True
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            flag = False
            break
    return flag



def plot_curve(x, data, title, save_path, name, xlabel="order", ylabel="mean v(S)"):
    plt.plot(x, data)
    plt.xlabel(xlabel, fontsize=FONT)
    plt.ylabel(ylabel, fontsize=FONT)
    plt.title(title, fontsize=FONT)
    plt.tick_params(labelsize=FONT)

    plt.tight_layout()
    mkdir(save_path)
    plt.savefig(os.path.join(save_path, f"{name}.png"))
    plt.close()


def plot_multiple_curves(x, data,
                         save_root, save_name, xlabel=None, ylabel=None,
                         font=20, linewidth=5, xticks=None, figsize=(6,4),
                         y_lim_low=None,
                         std_bars=None):
    # if type(data) == np.ndarray:
    #     data = [data]

    # fig = plt.figure(figsize=figsize, dpi=100)
    for i, arr in enumerate(data):
        l = plt.plot(x, arr, linewidth=linewidth)
        if std_bars is not None:  # shadow with +- std
            plt.fill_between(x, arr - std_bars[i], arr + std_bars[i],
                            facecolor=l[0].get_color(), alpha=0.2)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=font)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font)
    plt.tick_params(labelsize=font)
    if xticks is not None:
        plt.xticks(xticks, xticks)
    plt.ylim(y_lim_low, None)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f"{save_name}.png"))
    plt.savefig(os.path.join(save_root, f"{save_name}.svg"))
    plt.close()



###########################################################
###### utility functions for computing mean u(S) and std of u(S)
###########################################################

def compute_mean_vS(all_vS, num_players):
    mean_vS = []
    count = 0
    for order in range(0, num_players+1):
        num = math.comb(num_players, order)
        sum = 0
        for index in range(count, count + num):
            sum = sum + all_vS[index]
        mean = sum / num
        count = count + num
        mean_vS.append(mean)
    return mean_vS


def compute_std_vS(all_vS, num_players):
    std_vS = []
    count = 0
    for order in range(0, num_players+1):
        num = math.comb(num_players, order)
        uS_list_this_order = []
        for index in range(count, count + num):
            uS_list_this_order.append(all_vS[index])
        std = np.std(uS_list_this_order)
        count = count + num
        std_vS.append(std)
    return std_vS


def compute_mean_IS_strength(all_IS, num_players):
    mean_IS_strength_list = []
    count = 0
    for order in range(0, num_players+1):
        num = math.comb(num_players, order)
        all_IS_this_order = all_IS[count : count + num]
        mean_IS_strength_this_order = np.abs(all_IS_this_order).mean()
        assert mean_IS_strength_this_order >= 0
        count = count + num
        mean_IS_strength_list.append(mean_IS_strength_this_order)
    return mean_IS_strength_list


def compute_std_IS_strength(all_IS, num_players):
    std_IS_strength_list = []
    count = 0
    for order in range(0, num_players+1):
        num = math.comb(num_players, order)
        all_IS_this_order = all_IS[count : count + num]
        std_IS_strength_this_order = np.abs(all_IS_this_order).std()
        assert std_IS_strength_this_order >= 0
        count = count + num
        std_IS_strength_list.append(std_IS_strength_this_order)
    return std_IS_strength_list


###########################################################
###### utility functions for computing the bound of R^(k)
###########################################################

def compute_p_exact(arr, num_players):
    p_list = []
    for order1 in range(1, num_players + 1):
        for order2 in range(1, order1):
            p = math.log(arr[order2] / arr[order1], (order2 / order1))
            p_list.append(p)
    return max(p_list)


def compute_p_ref_order(arr, num_players, ref_order):
    p_list = []
    for order in range(1, num_players + 1):
        if order != ref_order:
            p = math.log(arr[order] / arr[ref_order], (order / ref_order))
            p_list.append(p)
    return max(p_list)


def compute_eta_k(IS_order_k):
    return np.sum(IS_order_k) / np.sum(np.abs(IS_order_k))


def to_n_ary(num, n):
    assert num > 0 and type(n)==int
    residual = num - int(num)
    num_for_n_ary = int(num)

    digit_list = np.arange(0, n)
    coef_list = []
    if num_for_n_ary == 0:
        coef_list = [0]
    else:
        while num_for_n_ary > 0:
            coef_list.append(digit_list[num_for_n_ary % n])
            num_for_n_ary = num_for_n_ary // n
    coef_list[0] += residual
    return np.array(coef_list) # start from 0 degree


class DeltakTooSmallException(Exception):
    pass

class DeltakEmptyException(Exception):
    pass

class LambdaCloseToZeroException(Exception):
    pass

class LambdaCloseToZeroWarning(Warning):
    pass

def compute_statistics_all_orders(all_IS, all_uS, num_players, p):
    assert all_uS[0] == 0
    p_floor = int(np.floor(p))
    IS_order1 = all_IS[1:num_players+1]
    uS_order1 = all_uS[1:num_players+1]
    # vEmpty = rewards[0]
    assert IS_order1.shape == uS_order1.shape == (num_players,)

    bar_u1 = np.mean(uS_order1)

    count = 0
    A_k_dict = {}
    A_k_sign_dict = {}
    eta_k_dict = {}
    q_k_dict = {}
    all_a_lt_p_floor_dict = {} # up to degree=p_floor
    delta_k_dict = {}
    lambda_k_dict = {}
    for k in range(0, num_players + 1): # order k
        num = math.comb(num_players, k)
        IS_order_k = all_IS[count : count + num]
        count = count + num

        if k == 0: # k=0, only have one interaction, skip it
            continue

        A_k = np.sum(IS_order_k)
        A_k_dict[k] = A_k

        eta_k = compute_eta_k(IS_order_k)
        eta_k_dict[k] = eta_k

        sign = np.sign(A_k / bar_u1)
        A_k_sign_dict[k] = sign

        abs_ratio_A_k_bar_u1 = np.abs(A_k / bar_u1)

        if k == 1: # a trick to avoid empty delta_k_list
            abs_ratio_A_k_bar_u1 = num_players

        coef_list = to_n_ary(abs_ratio_A_k_bar_u1, n=num_players) # array
        q_k = len(coef_list) - 1
        q_k_dict[k] = q_k

        if q_k <= p_floor - 1:
            a_lt_p_floor_list = np.pad(coef_list,
                            pad_width=(0, p_floor - len(coef_list)),
                            mode='constant',
                            constant_values=0) # degree from 0 to p_floor-1
            assert len(a_lt_p_floor_list) == p_floor
            all_a_lt_p_floor_dict[k] = sign * a_lt_p_floor_list # important! multiply by the sign

        else:
            a_lt_p_floor_list = coef_list[:p_floor]
            assert len(a_lt_p_floor_list) == p_floor
            all_a_lt_p_floor_dict[k] = sign * a_lt_p_floor_list # important! multiply by the sign

            sum_ge_p_floor = 0
            for i in np.arange(p_floor, q_k + 1): # p_floor <= i <= q_k, i is the degree
                sum_ge_p_floor += coef_list[i] * (num_players ** i)

            delta_k = math.log(sum_ge_p_floor, num_players) - p

            delta_k_dict[k] = delta_k

    assert len(A_k_dict) == len(A_k_sign_dict) == len(eta_k_dict) == len(all_a_lt_p_floor_dict) == num_players

    if len(delta_k_dict) == 0:
        raise DeltakEmptyException("delta_k_dict is empty")

    delta = max(delta_k_dict.values())
    for k in range(1, num_players + 1): # order 0 is skipped
        if k in delta_k_dict:
            delta_k = delta_k_dict[k]
            lambda_k_dict[k] = A_k_sign_dict[k] * (num_players ** (delta_k - delta))
        else:
            lambda_k_dict[k] = 0

    return  A_k_dict, \
            A_k_sign_dict, \
            eta_k_dict, \
            q_k_dict, \
            all_a_lt_p_floor_dict, \
            delta_k_dict, \
            lambda_k_dict, \
            bar_u1


def compute_comb_ratio_weighted_sum(value_dict, n, m0, M):
    assert n >= M and m0 >= M
    # assert len(value_dict) == M
    sum = 0
    for k in range(1, M + 1): # 1 to M
        sum += math.comb(m0, k) / math.comb(n, k) * value_dict[k]
    return sum



def compute_delta_bound(lambda_k_dict, all_a_lt_p_floor_dict, p, p_floor, num_players, M):
    # compute the value of lambda, and the bound of delta
    assert len(lambda_k_dict) == num_players
    m0_best = 0
    delta_bound_best = np.inf
    for m0 in range(max(M, num_players - M), num_players + 1): # trick: m0 cannot be smaller than M, otherwise the combination number cannot be computed
        lambda_tmp = compute_comb_ratio_weighted_sum(lambda_k_dict, n=num_players, m0=m0, M=M)
        if abs(lambda_tmp) < 1e-8:
            continue

        sum_a_i_div_n = 0
        for i in range(0, p_floor): # 0 to p_floor-1
            a_i_all_orders_dict = {} # a_i^(k), k=1 ... n
            for k, a_lt_p_floor_list in all_a_lt_p_floor_dict.items():
                assert len(a_lt_p_floor_list) == p_floor
                a_i_all_orders_dict[k] = a_lt_p_floor_list[i]
            assert len(a_i_all_orders_dict) == num_players
            a_i = compute_comb_ratio_weighted_sum(a_i_all_orders_dict, n=num_players, m0=m0, M=M)
            sum_a_i_div_n += a_i * (num_players ** (i - p))

        if lambda_tmp > 0:
            denominator = 1 - sum_a_i_div_n
            delta_bound = math.log(denominator / lambda_tmp, num_players)
        elif lambda_tmp < 0:
            denominator = sum_a_i_div_n
            delta_bound = math.log(denominator / (-lambda_tmp), num_players)
        else:
            raise ValueError("delta_bound divided by 0")

        if delta_bound < delta_bound_best:
            delta_bound_best = delta_bound
            m0_best = m0

    if delta_bound_best == np.inf:
        raise LambdaCloseToZeroException(f"lambdas under different m0 are all close to 0")
    return delta_bound_best, m0_best


def compute_real_R_k(all_IS, num_players, tau):
    real_R_k_dict = {}
    count = 0
    for k in range(0, num_players + 1):  # order k
        num = math.comb(num_players, k)
        IS_order_k = all_IS[count: count + num]
        count = count + num

        if k == 0: # k=0, only have one interaction, skip it
            continue

        real_R_k = np.sum(np.abs(IS_order_k) >= tau)
        real_R_k_dict[k] = real_R_k
    return real_R_k_dict
