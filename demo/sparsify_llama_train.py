import torch

import os
Project_root = os.getcwd()
import sys
sys.path.append(Project_root)

from interaction.data import *
from interaction.models import CalculatorLlama
from interaction.harsanyi.calculate import Harsanyi
from interaction.baseline import load_baseline_embeds

import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir_name', type=str, default="sparsify_result_llama")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--qthres', type=float, default=0.04)
    parser.add_argument('--sparsify_loss', type=str, default="l1")
    parser.add_argument('--mode', type=str, default="pq")
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()

    assert args.num_samples <= 1000, "max number of testing samples is 1000"

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    calculator = CalculatorLlama()
    calculator = calculator.to(device)
    calculator.eval()

    # prepare the config
    config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "qthres": args.qthres,
        "reward_function": "gt-log-odds-v0",
        "sparsify_loss": args.sparsify_loss,
        "q_tricks": True,
        "piece_wise": False,
        "mode": args.mode
    }

    # prepare the Harsanyi calculator
    harsanyi = Harsanyi(Calculator=calculator, config=config)
    
    data_path = "data/llama_data"
    baseline_path = "baseline_results/llama_baseline"

    save_note = config["reward_function"] + f"_mode={config['mode']}" \
                                            f"_qthres={config['qthres']}" \
                                            f"_lr={config['lr']}" \
                                            f"_epoch={config['epochs']}"

    save_dir = os.path.join(f"results/{args.save_dir_name}", save_note)
    sen_path = os.path.join(data_path, "sentence.txt")
    word_path = os.path.join(data_path, "player_words.json")
    id_path = os.path.join(data_path, "player.json")


    baseline = load_baseline_embeds(baseline_path)
    
    # get data
    with_indent = False
    sentences = load_sentences(sen_path,with_indent)
    if not os.path.exists(id_path):
        player_words = load_player_words(word_path)
        player_ids = get_player_ids_from_word(calculator.tokenizer,sentences,player_words,with_indent,save_dir=data_path)
    else:
        player_ids = load_player_ids(id_path)
    data_loader = PlayerDataset(sentences,player_ids)

    for i, (sentences, player_ids) in enumerate(data_loader):
        if i >= args.num_samples:
            continue
        save_path = os.path.join(save_dir, f"sample{i}")
        input_ids = calculator.tokenizer(sentences,return_tensors="pt")['input_ids']
        results = harsanyi(input_ids, player_ids, baseline, save_path, calculator.tokenizer)
    
        # Check the interaction
        label = results['label']
        masks = results['masks'].cpu()
        I_and = results['I_and'].cpu()
        I_or = results['I_or'].cpu()
        rewards = results['rewards'].cpu()

        # save
        np.save(os.path.join(save_path, 'masks'), masks.numpy())
        np.save(os.path.join(save_path, 'Iand'), I_and.numpy())
        np.save(os.path.join(save_path, 'Ior'), I_or.numpy())
        np.save(os.path.join(save_path, 'rewards'), rewards.numpy())