import torch.nn as nn
import torch
import os
import json

class Calculator():
    '''class to obtain the model output logits from input embeddings, ids, sentences
    params:
    ==========

    model: nn.Module
        original trained generative model
    tokenizer: transformer.Tokenizer
    config: dict
        config dict to decide the different attri.
    '''
    def __init__(self, ) -> None:
        self.cal_config = self.load_config()

    def load_config(self):
        model_dir = os.path.dirname(__file__)
        config_path = os.path.join(model_dir,'cal_config.json')
        with open(config_path,'r') as f:
            cal_config = json.load(f)
        return cal_config

    def get_layer(self, 
                  model: nn.Module,
                  layer_name: str,
        ) -> nn.Module:

        layer_list = layer_name.split("/")
        prev_module = model
        for layer in layer_list:
            prev_module = prev_module._modules[layer]

        return prev_module

    def get_embeds(self, 
                   input_ids: torch.Tensor, ):
        with torch.no_grad():
            word_embeddings = self.word_embed(input_ids)
        return word_embeddings

    def forward(self,
                input_ids = None,
                input_embeds = None,
                input_sentences = None,
        ) -> torch.Tensor:
        return None