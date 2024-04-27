# CDnet
## Introduction
Graph neural networks have shown their advantages across various real-world graph data. Initially, the success of graph neural networks mainly relied on message passing methods that focus on local information. Subsequently, methods focusing on global information through graph diffusion were proposed, effectively overcoming the limitations imposed by relying on direct neighbors ("information islands"). However, balancing the consideration of local and global information in the graph and achieving efficient node classification tasks for both homogenous and heterogeneous graphs remains a challenge. We propose a Constrained Diffusion network model (CDnet) designed to optimize graph diffusion efficiently, targeting minimal yet effective diffusion scopes, thereby efficiently acquiring global information about the graph. Additionally, the model integrates feature information flow routing, guiding the diffusion process for each node and addressing the traditional shortcomings of graph diffusion neural networks that rely solely on structure. We demonstrate that the graph diffusion process of CDnet, under the routing guide, enables significant improvements in the performance of graph node classification tasks on multiple experimental datasets while substantially compressing the scale of graph diffusion. Notably, for nodes with weaker structural roles in the graph, CDnet significantly improves classification outcomes compared to other models.
![image](https://github.com/FangjingLi/CDnet/assets/157601218/ce002d29-00b0-4196-ae3a-8b65e772849f)
**Figure 1:** The overall framework of CDnet.
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

To train and evaluate CDnet for node classification with the command line:

```python
python CDnet.py --dataset_name Cora --a 0.97 --max_epochs 10000 --process_feature cos --architecture GCN --K0_mul 0.5
```

