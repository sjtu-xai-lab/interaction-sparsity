import sys
import os
import json
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertTokenizer

from .modify_aquila import ModifiedAquila
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    return ModifiedAquila.from_pretrain(download_path=model_path,  model_name='Aquila-7B', only_download_config=False)

def load_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path)

class CalculatorAquila(nn.Module):
    """
    Calculator model for the bert_large
    Input the input_ids or the input_embeds or the input_sentences
    Output the logit of the predicted word
    Can get word embed given the input_ids
    """
    def __init__(self,):
        super().__init__()

        self.cal_config = self.load_cal_config()
        self.cal_model = load_model(self.cal_config["model_path"])
        self.tokenizer = load_tokenizer(self.cal_config["tokenizer_path"])
        self.word_embed = self.get_layer(self.cal_config["word_embed"])

    def load_cal_config(self):
        model_dir = os.path.dirname(__file__)
        config_path = os.path.join(model_dir,'cal_config.json')
        with open(config_path,'r') as f:
            cal_config = json.load(f)
        return cal_config
    
    def get_layer(self, 
                  layer_name: str,
        ) -> nn.Module:
        layer_list = layer_name.split("/")
        prev_module = self.cal_model
        for layer in layer_list:
            prev_module = prev_module._modules[layer]

        return prev_module

    def get_embeds(self, 
                   input_ids: torch.Tensor, ):
        with torch.no_grad():
            word_embeddings = self.word_embed(input_ids)
        return word_embeddings

    def forward(self,
                input_ids: torch.LongTensor = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                input_sentences: str = None,
                ) -> torch.Tensor:
        
        if input_sentences is not None:
            input_ids = self.tokenizer(input_sentences,return_tensors="pt")['input_ids'].cuda()
        
        outputs = self.cal_model(input_ids,inputs_embeds)
        return outputs



