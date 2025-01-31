# CDGConv

## Installation

The requirement.txt included all the dependencies. lt main depends on:

- python=3.8.10
- torch=2.0.1+cu118
- torch-geometric=2.3.1
- torch-scatter=2.1.1
- torch-sparse=0.6.17
- torchvision=0.15.2+cu118
- scipy=1.7.3
- seaborn=0.13.2
- pyyaml=6.0.1
- numpy=1.22.4
- pandas=1.5.3
- scikit-learn=1.2.2
- matplotlib=3.6.1

CUDA and Torch official installation reference:

https://developer.nvidia.cn/

https://pytorch.org/

## Run the code

To train and evaluate CGDConv for node classification with the command line:

```python
python CGDC.py --dataset_name Cora --a 0.97 --max_epochs 10000 --process_feature cos --architecture GCN --K0_mul 0.5
```

