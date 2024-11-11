# Interaction-sparsity
PyTorch implementation of the paper "Where We Have Arrived in Proving the Emergence of Sparse Interaction Primitives in AI Models" in ICLR 2024 ([paper](https://openreview.net/forum?id=3pWSL8My6B)).

## Requirements

- Python 3.8.0
- pytorch 2.0.1
- CUDA 11.7
- numpy 1.24.4
- transformers 4.31.0

All models were tested on a single A100 GPU.

You can also try the following command to install dependencies:
```
conda env create -f environment.yml
```

## Usage

To better reproduce our results in the paper, **we suggest directly downloading all the raw interactions** from this [Google Drive](https://drive.google.com/drive/folders/1lRq3TR1U7_3ijw_30-nM1ucRJynkXh9E?usp=sharing).

To obtain the box diagram in Figure 4, run the following command. Fill the `--model` argument with one of [`opt`, `llama`, `aquila`].
```
python ./demo/plot_inter_strength_boxplot.py --model=opt
```

To visualize the monotonicity assumption in Figure 5(a), run the following command.
```
python ./demo/plot_monotonicity_examples.py --model=opt
```

To reproduce the statistics in Table 1, run the following command.
```
python ./demo/check_monotonicity.py --model=opt
python ./demo/count_salient_concepts.py --model=opt
```

To reproduce the statistics in Table 2 and Figure 5(b), run the following command.
```
python ./demo/compute_p_and_bound.py --model=opt
```

## Project Page
See our project page [here](https://sjtu-xai-lab.github.io/InteractionSparsity/)!


## Citation

~~~late
@inproceedings{
  ren2024where,
  title={Where We Have Arrived in Proving the Emergence of Sparse Interaction Primitives in {DNN}s},
  author={Qihan Ren and Jiayang Gao and Wen Shen and Quanshi Zhang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}


