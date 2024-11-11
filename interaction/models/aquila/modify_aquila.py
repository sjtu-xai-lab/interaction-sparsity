from transformers import BertForMaskedLM
from typing import List, Optional, Tuple, Union
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.model.aquila_model import *
import torch

class ModifiedAquila(AQUILAModel):
    """
    Modfied model for the bert-large-acsed
    Input the input_ids or the input_embeds
    Output the logit of the predicted word
    """
    def __init__(self, config):
        super().__init__(config)
    
    # def from_pretrained(self, model_path):
    #     config_file = os.path.join(model, 'config.json')
    #     model = AQUILAModel.init_from_json(config_file=config_file)

    #     checkpoint_path = os.path.join(model, "pytorch_model.bin")
    #     model.load_weights(checkpoint_path)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_pos=0, labels=None, **kwargs
        ) -> torch.Tensor:
        
        if input_ids is not None:
            _bsz, seqlen = input_ids.shape
            h = self.tok_embeddings(input_ids)
            device = input_ids.device
        else:
            _bsz, seqlen, _zzz = inputs_embeds.shape
            h = inputs_embeds
            device = inputs_embeds.device
            
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        self.start_pos = start_pos
        if self.config.checkpoint_activations:
            for layer in self.layers:
                layer.use_cache = self.use_cache
                layer.start_pos = start_pos
                h = checkpoint(create_custom_forward(layer), h, freqs_cis, mask)
        elif os.getenv("ENV_TYPE") == "bmtrain" and self.config.bmt_comm_overlap:
            # to overlap communication with computation
            for layer in self.layers:
                layer.use_cache = self.use_cache
                layer.start_pos = start_pos
                
            h = self.layers(h, freqs_cis, mask)
        else:
            for layer in self.layers:
                layer.use_cache = self.use_cache
                layer.start_pos = start_pos
                h = layer(h, freqs_cis, mask)
                
        h = self.norm(h)
        if labels is not None:
            h = self.output(h)

            shift_logits = h[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            bsz_half = _bsz // 2
            bsz_split = bsz_half * seqlen
            bsz_total = _bsz * seqlen
            ## torch 1.12.1 
            ## CUDA Illegal memory access on CrossEntropyLoss with large batch size
            ## https://github.com/pytorch/pytorch/issues/85005
            if self.config.fix_large_bsz and bsz_total > bsz_split:
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1).long()
                loss_split = self.loss_func(shift_logits[:bsz_split, :], shift_labels[:bsz_split]).mean()
                loss_remain = self.loss_func(shift_logits[bsz_split:, :], shift_labels[bsz_split:]).mean()
                bsz_remain = bsz_total - bsz_split
                ## NaN
                #loss = (loss_split * bsz_split + loss_remain * bsz_remain) / bsz_total
                loss = bsz_split / bsz_total * loss_split + bsz_remain / bsz_total * loss_remain
            else:
                loss = self.loss_func(
                    shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1).long()).mean()
            
            return {
                'logits': h, 
                'loss': loss,
                'hidden_states': h,
            }
        else :
            output = self.output(h[:, -1, :])  # only compute last logits
            return output.float()



if __name__ == '__main__':
    pass