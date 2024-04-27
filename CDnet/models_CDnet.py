__author__ = "Stefan Weißenberger and Johannes Gasteiger"
__license__ = "MIT"

from itertools import combinations
from typing import List

import numpy as np
import torch
from torch.nn import ModuleList, Dropout, ReLU, BatchNorm1d, Linear, Parameter, init
from torch_geometric.nn import GCNConv, GATv2Conv, MessagePassing
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor


class GCN(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 a: float = 0.1,
                 k: int = 16,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GCN, self).__init__()

        self.k=k
        self.a=a
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes] #计算了输入特征的维度、隐藏层的维度和输出的类别数，并存储在 num_features 列表中
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]): #zip(num_features[:-1], num_features[1:])作用是将num_features列表中相邻的元素配对在一起
            layers.append(GCNConv(in_features, out_features,edge_dim=1))
        self.layers = ModuleList(layers) #将layers列表包装在PyTorch的ModuleList中，以将它们作为模型的一部分

        self.reg_params = list(layers[0].parameters()) #创建一个列表 self.reg_params，其中包含了第一层 GCNConv 层的参数。这些参数可能用于正则化
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()]) #创建一个列表 self.non_reg_params，其中包含了除第一层外的所有 GCNConv 层的参数。这些参数可能不用于正则化

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()



    def reset_parameters(self): #自定义方法，用于重置模型中每个层的参数
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data, G2_edge_attr,G1_edge_attr_matrix,G3_edge_index, G3_edge_attr):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, edge_index, edge_attr = data.x.to(device), G3_edge_index, G3_edge_attr
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_attr)#在每一层的循环中，输入节点特征 x 通过 GCNConv 层进行图卷积操作

            if i == len(self.layers) - 1:#检查当前层是否为最后一层
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)  # 模型的输出 x 通过 log_softmax 函数，以获得类别的对数概率分布。通常用于多类别分类问题。


class GAT(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 a: float = 0.1,
                 k: int = 16,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GAT, self).__init__()

        self.k=k
        self.a=a
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes] #计算了输入特征的维度、隐藏层的维度和输出的类别数，并存储在 num_features 列表中
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]): #zip(num_features[:-1], num_features[1:])作用是将num_features列表中相邻的元素配对在一起
            layers.append(GATv2Conv(in_features, out_features,edge_dim=1))
        self.layers = ModuleList(layers) #将layers列表包装在PyTorch的ModuleList中，以将它们作为模型的一部分

        self.reg_params = list(layers[0].parameters()) #创建一个列表 self.reg_params，其中包含了第一层 GCNConv 层的参数。这些参数可能用于正则化
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()]) #创建一个列表 self.non_reg_params，其中包含了除第一层外的所有 GCNConv 层的参数。这些参数可能不用于正则化

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self): #自定义方法，用于重置模型中每个层的参数
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data, G2_edge_attr,G1_edge_attr_matrix,G3_edge_index, G3_edge_attr):

        x, edge_index, edge_attr = data.x, G3_edge_index, G3_edge_attr
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr)#在每一层的循环中，输入节点特征 x 通过 GCNConv 层进行图卷积操作

            if i == len(self.layers) - 1:#检查当前层是否为最后一层
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1) # 模型的输出 x 通过 log_softmax 函数，以获得类别的对数概率分布。通常用于多类别分类问题。

class MLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels,a,
                 num_layers: int=2,
                 dropout: float=0.5):
        super(MLP, self).__init__()
        self.a=a
        self.lins = ModuleList()
        self.bns = ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(Linear(in_channels, out_channels))
        else:
            self.lins.append(Linear(in_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(Linear(hidden_channels, hidden_channels))
                self.bns.append(BatchNorm1d(hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
    # def forward(self, data):
        if not input_tensor:
            x = data.x
        else:
            x = data
        # x = data.x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=1)
