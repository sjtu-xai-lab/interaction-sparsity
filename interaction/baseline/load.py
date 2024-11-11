import os

import numpy as np
import torch

def load_baseline_embeds(
        embeds_path: str = None,
    ) -> torch.Tensor:
    embeds_path = os.path.join(embeds_path,'baseline_embeds_list.npy')
    baseline_embeds_list = np.load(embeds_path)
    baseline_embeds = torch.tensor(baseline_embeds_list[-1],dtype=torch.float32)
    
    return baseline_embeds
