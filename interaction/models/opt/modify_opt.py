from transformers import OPTForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Union
import torch

class ModifiedOpt(OPTForCausalLM):
    """
    Modfied model for the bert-large-acsed
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )

        logits = self.lm_head(outputs[0]).contiguous()[:,-1]

        return logits


if __name__ == '__main__':
    pass