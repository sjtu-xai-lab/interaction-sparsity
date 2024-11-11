import os
import numpy as np
import pandas as pd
import json

from torch.utils.data import Dataset


class PlayerDataset(Dataset):
    """
    Given sentences and player ids, get the player dataset


    Parameters
    ----------
    sentences: list[strings1,strings2, ... stringsn]
                list of unencoded sentences
    player_ids: list[[Tensor,Tensor,Tensor,...,Tensor],[Tensor,Tensor],...]
                list of player word id, index begins with 0, if set None, 
                the Dataset will create valid player ids

    Return
    ---------
    dataset: torch.utils.data.Dataset
                dataset of the sentences and player_ids 
    """
    def __init__(self, 
                 sentences: list, 
                 player_ids: list,
                 ):
        super(PlayerDataset, self).__init__()
        
        self.sentences = sentences
        self.player_ids = tuple(player_ids)
        self.count = -1
    
    def next(self):
        self.count + 1
        return self[self.count % len(self)]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (self.sentences[idx]), self.player_ids[idx]


if __name__ == '__main__':
    pass