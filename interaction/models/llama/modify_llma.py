from transformers import LlamaForCausalLM,LlamaTokenizer
from typing import List, Optional, Tuple, Union
import torch

class ModifiedLlama(LlamaForCausalLM):
    """
    Modfied model for the lamma
    Input the input_ids or the input_embeds
    Output the logit of the predicted word
    """
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        ) -> torch.Tensor:
        
        if input_ids is not None:
            attention_mask = torch.ones_like(input_ids).to(self.device)
        if inputs_embeds is not None:
            attention_mask = torch.ones_like(inputs_embeds[:,:,0],dtype=torch.int64).to(self.device)

        output_attentions = self.config.output_attentions
        output_hidden_states = (self.config.output_hidden_states)

        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=None,
            )

        hidden_states = outputs[0]
        prediction_scores = self.lm_head(hidden_states)[:,-1]
        return prediction_scores

if __name__ == '__main__':
    pass