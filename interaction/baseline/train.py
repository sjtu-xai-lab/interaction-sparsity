import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm


from interaction.baseline.utils import generate_mask, plot_simple_line_chart, sample_subset
from interaction.baseline.hook import Hook


def _get_batch_output(calculator: nn.Module,
                      input_ids: torch.Tensor,
                      player_ids: torch.Tensor,
                      baseline_embeds: torch.Tensor,
                      loss_tag: str,
                      num_subset_sample: int,
                      num_subsets: int,
                      subset_unions:list,
                      device: str,):
    '''function to obtain batch output for masked samples 
    for each sample, there are in total 40 different masked status.
    
    params
    =======
    calculator: nn.Module
            modified model to make inference on masked samples (masks on token embeddings)
    input_ids: Tensor
            model input
    baseline_embeds: Tensor
            baseline values to mask samples 
    num_subset_sample: int       
    device: str
    subset_unions: list[list]
            union index, e.g. [0,1,2] stands for S_1 U S_2 U S_3
                              [0,2] stands for S_1 U S_3

    return 
    =======
    values: Tensor shape (args.num_subset_sample, num_unions)
            [v(x_{S_1 ∪ S_2 ∪S_3}), v(x_{S_1 ∪ S_3}), v(x_{S_2 ∪ S_3}), v(x_S_3)] for each masked sample
    '''
    ### the length of the sentence
    assert len(input_ids) == 1
    input_len = len(input_ids[0])
    num_unions = len(subset_unions)
    
    ### for each sample, we will have args.num_subset_sample * len(subset_unions) 
    ### different masked samples, e.g., 10*4 = 40
    total_size = num_subset_sample * num_unions
    input_ids_batch = input_ids.repeat(total_size,1)
    mask_batch = np.zeros(shape=(total_size, input_len, 1),dtype=np.float32)
    
    # generate masks
    for ind in range(num_subset_sample):
        subsets = sample_subset(player_ids,num_subsets)
        subset_masks = generate_mask(input_len,player_ids,subsets,subset_unions,device)
        subset_masks = np.expand_dims(subset_masks,axis=-1)
        mask_batch[ind*num_unions:(ind+1)*num_unions] = subset_masks
    


    # forward the model
    mask_batch = torch.tensor(mask_batch,dtype=torch.float32).to(device)
    inputs_embeds_batch = calculator.get_embeds(input_ids_batch)
    masked_embeds_batch = mask_batch * inputs_embeds_batch + (1-mask_batch) * baseline_embeds
    scores = calculator(inputs_embeds = masked_embeds_batch)

    # if output the mid layer just return
    if loss_tag != 'raw':
        return None

    # find gt
    with torch.no_grad():
        gt_scores = calculator(input_ids=input_ids)
        gt = gt_scores.argmax(dim=1).item()
    # calcuate v(S) = log(p/(1-p))
    outputs_prob = torch.softmax(scores, dim=1)
    eps = 1e-7
    values = outputs_prob[:, gt]
    values = torch.log(values / (1 - values + eps)+eps)
    
    # reshape the value to (args.num_subset_sample, num_unions)
    values = values.reshape(num_subset_sample, num_unions)
    return values


def train_baseline_embeds(calculator = None,
                          data_loader = None,
                          init_baseline_id: int = None,
                          out_path: str = None,
                          loss_tag: str = None,
                          device: str = None,
                          norm_limit = 1.8,
                          epoch: int = 200,
                          lr: float = 1e-3,
                          num_subset_sample: int = 10,
                          num_subsets: int = 3,
                          subset_unions:list=[[0,1,2],[0,2],[1,2],[2]]
                          ):
    '''
    function to optimize baseline values,
    params
    =======
    data_loader: DataLoader
    calculator: nn.Module
        modified model to obtain the output via masked embeddings
    num_subset_sample: int,
    lr: int,
        learning rate, usually 1e-3
    epoch: int,
        training epoch, usually 100
    out_path: str,
        path to save the results
    device: str,

    return
    ======
    baseline_embeds: torch.Tensor
        baseline embeddings            
    '''
    assert loss_tag in ["L1","L2","raw"]

    if loss_tag == "raw":
        hook = None
    else:
        mid_layer_name = calculator.cal_config["mid_layer_name"]
        hook = Hook(calculator.cal_model,[mid_layer_name])

    # define the trainable parameters
    baseline_id = torch.tensor(init_baseline_id,dtype=torch.int64).unsqueeze(0).to(device)
    init_baseline_embeds = calculator.get_embeds(baseline_id)
    baseline_embeds = Variable(init_baseline_embeds, requires_grad=True)
    optimizer = torch.optim.SGD([baseline_embeds], lr=lr, momentum=0.9)

    losses = [] 
    baseline_norms = []
    baseline_embeds_list = [init_baseline_embeds.detach().cpu()] 
    
    #test cude out of memory
    def test_mem():
        allocated_memory = torch.cuda.memory_allocated()
        cached_memory = torch.cuda.memory_reserved()
        print(f"Allocated memory: {allocated_memory / (1024**3):.2f} GB, Cached memory: {cached_memory / (1024**3):.2f} GB")
    
    # start training
    pbar = tqdm(range(epoch), desc="Optimizing baseline_embeds", ncols=100)
    for it in pbar:
        running_loss = 0.
        for ind, (sentences, player_ids) in tqdm(enumerate(data_loader)):
            input_ids = calculator.tokenizer(sentences, return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(device)
            # get model output for all masked status on the target sample
            torch.cuda.empty_cache()
            values = _get_batch_output(
                calculator=calculator,
                input_ids=input_ids,
                player_ids=player_ids,
                baseline_embeds=baseline_embeds,
                loss_tag = loss_tag,
                num_subset_sample=num_subset_sample,
                num_subsets=num_subsets,
                subset_unions=subset_unions,
                device=device,)
            if loss_tag == 'raw':
            # loss = E_{S_1,S_2, S_3}[|v(x_{S_1 ∪ S_2 ∪S_3})−[v(x_{S_1 ∪ S_3})+v(x_{S_2 ∪ S_3})]+v(x_S_3)|]
                loss = torch.abs(values[:,0] - (values[:,1] +values[:,2]) + values[:,3])
            else:
                features = hook.get_features(mid_layer_name)
                features = features.reshape(num_subset_sample, len(subset_unions),len(input_ids[0]),-1)
                values = features[:,0] - (features[:,1] +features[:,2]) + features[:,3]
                if loss_tag == 'L1':
                    loss = torch.sum(torch.abs(values), dim=1)
                elif loss_tag == "L2":
                    loss = torch.sum(values**2, dim=1)
            # optimize
            loss = torch.sum(loss) / num_subset_sample
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
            running_loss += loss.item()
            #limit baseline_embeds norm
            # with torch.no_grad():
            #     baseline_norm = torch.norm(baseline_embeds, p=2).item()
            #     if baseline_norm > norm_limit:
            #         baseline_embeds = baseline_embeds / baseline_norm * norm_limit

        # compute loss over all samples 
        average_loss = running_loss / len(data_loader)
        print(f'loss for epoch {it +1} is {average_loss}\n')
        pbar.set_postfix_str(f"loss={average_loss:.4f}")
        
        #save results
        losses.append(average_loss)
        with torch.no_grad():
            basline_norm = torch.norm(baseline_embeds,p=2).item()
            baseline_norms.append(basline_norm)
        baseline_embeds_list.append(baseline_embeds.detach().cpu())
        # plot loss 
        plot_simple_line_chart(data=losses, xlabel="epoch", ylabel=f"loss", 
                                title="", save_folder=out_path, 
                                save_name=f"loss_curve_optimize_baseline")
        plot_simple_line_chart(data=baseline_norms, xlabel="epoch", ylabel=f"baseline_norm", 
                                title="", save_folder=out_path, 
                                save_name=f"baseline_norm_curve_optimize_baseline")
        # save baseline values
        np.save(os.path.join(out_path, 'baseline_embeds_list.npy'),torch.cat(baseline_embeds_list,axis=0).numpy())
        np.save(os.path.join(out_path, 'losses.npy'), losses)
        np.save(os.path.join(out_path, 'baseline_norms.npy'), baseline_norms)

    return baseline_embeds.detach()