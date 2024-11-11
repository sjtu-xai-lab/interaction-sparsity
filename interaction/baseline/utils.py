import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt


def setup_seed(seed=1029):
    print(f"set seed {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def sample_subset(player_ids: np.ndarray,
                  num_subsets: int,
                  ):
    '''function to randomly sample subsets
    Parameters
    ----------
    players_ids: [[1],[4],[2]]
                list of player ids
    subset_sizes: list
                list of size of the size of each subset, 
                the size of first two subsets should be in range [2,3]
                the size of the last subset take the minimun between the 
                default number and the remaining number of players

    Return
    ---------
    subsets: list[S1, S2, ..., Sn]
                list of the selected subsets
    '''
    subsets = []
    num_players = len(player_ids)
    player_pos = np.arange(num_players,dtype=np.int64)

    for ind in range(num_subsets-1):
        subset_size = np.random.randint(low=1, high=4)
        subset = np.random.choice(player_pos, subset_size, replace=False)
        # to ensure the sampled subsets are mutually disjoint
        player_pos = np.delete(player_pos, np.searchsorted(player_pos, subset))
        num_players = num_players - subset_size
        subsets.append(subset)
    
    # p = 1/3 取 subset_size = num_players
    subset_size_all = np.random.choice([True,False], p=[1/3, 2/3])
    if subset_size_all:
        subset_size = num_players
        subset = np.random.choice(player_pos, subset_size, replace=False)
        subsets.append(subset)
    
        return subsets
    
    # things going wrong
    subset_size = np.random.randint(low=0, high = num_players)
    subset = np.random.choice(player_pos, subset_size, replace=False)
    subsets.append(subset)
    return subsets

def get_mask_token_pos(player_ids,subset = None):
    mask_token_pos = []
    if subset is None:
        subset = np.arange(len(player_ids),dtype=np.int64)
    for idx in subset:
        mask_token_pos += player_ids[idx]
    if len(mask_token_pos) == 0:
        return None
    return np.array(mask_token_pos)

def generate_mask(input_len: int, 
                  player_ids: list,
                  subsets: list, 
                  subset_unions: list,
                  device: str,
                  )->np.ndarray:
    '''
    function to generate masks
    
    parameters
    ----------
    input: int
            the length of the input tokens
    player_pos: list
            list of player ids index
    subsets: list
            list of subsets [S1, S2, ..., Sn]
    subset_unions: list[list]
            list of subset unions, i.e., [0,1,2] = S1 U S2 U S3 

    Return
    mask: Tensor
        size (len(subset_unions), padding_length)
    '''
    ### initialize the mask with 0 
    mask = np.ones(shape=(len(subset_unions), input_len),dtype=np.float32)
    mask_token_pos = get_mask_token_pos(player_ids)
    mask[:,mask_token_pos] = 0

    ### the word in subset S remains the same while the word in N\S will
    ### be replaced by the baseline value. 
    for ind, union in enumerate(subset_unions):
        if len(union) > 1:
            _union = [subsets[u] for u in union]
            subset = np.concatenate(_union, axis=None) 
        else:
            subset = subsets[union[0]]
        mask_token_pos = get_mask_token_pos(player_ids,subset)

        if mask_token_pos is None:
            continue
        mask[ind, mask_token_pos] = 1

    return mask

def plot_simple_line_chart(data: list, 
                           xlabel: str, 
                           ylabel: str, 
                           title: str, 
                           save_folder: str, 
                           save_name: str, 
                           X=None):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if X is None: X = np.arange(len(data))
    plt.plot(X, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")

if __name__ == '__main__':
    pass