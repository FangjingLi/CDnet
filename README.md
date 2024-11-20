# CDnet
## Introduction
In the data-driven era, graph data is widely distributed in multiple fields, and its complex network relationships pose higher challenges to data analysis models. Although graph neural networks (GNNs) perform well in processing non-Euclidean spatial data, traditional methods are usually limited to capturing local information through message passing or relying on graph diffusion to obtain global information. This single perspective makes it difficult to achieve a balance between global and local information, especially in heterogeneous graphs, which can easily lead to information loss or over-smoothing. To this end, we propose Constrained Diffusion Graph Neural Networks (CDnet), which optimizes the graph diffusion range by adding dual constraints: Constrained Diffusion (CD) and Feature Information Flow Routing (FIFR). The CD module reduces the diffusion depth by adaptively reducing the number of diffusion steps, efficiently captures global information and reduces invalid propagation. The FIFR mechanism dynamically regulates the direction and strength of information flow between nodes to ensure precise control of information transmission and retention of local features. Extensive experimental results on public homogeneous and heterogeneous datasets show that CDnet outperforms state-of-the-art methods, demonstrating its excellent generalization and robustness in complex graph analysis.

<img width="1837" alt="image" src="https://github.com/user-attachments/assets/ee42af39-a544-4288-ba2e-7991aecef42f">

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

