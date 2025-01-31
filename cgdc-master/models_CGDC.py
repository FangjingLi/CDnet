__author__ = "Stefan Weißenberger and Johannes Gasteiger"
__license__ = "MIT"

from itertools import combinations
from typing import List

import numpy as np
import torch
from torch.nn import ModuleList, Dropout, ReLU, BatchNorm1d, Linear, Parameter, init
from torch_geometric.nn import GCNConv, GATv2Conv, MessagePassing, SAGEConv
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

        self.k = k
        self.a = a
        # The dimension of the input feature, the dimension of the hidden layer, and the number of categories of the output are calculated and stored in the num_features list.
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        # The function of zip(num_features[:-1], num_features[1:]) is to pair adjacent elements in the num_features list together.
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features, edge_dim=1))
        # Wrap the layers list in PyTorch's ModuleList to include them as part of the model.
        self.layers = ModuleList(layers)

        # Create a list self.reg_params that contains the parameters of the first GCNConv layer. These parameters may be used for regularization.
        self.reg_params = list(layers[0].parameters())
        # Create a list self.non_reg_params that contains the parameters of all GCNConv layers except the first one. These parameters may not be used for regularization.
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        # A custom method used to reset the parameters of each layer in the model.
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data, G2_edge_attr, G1_edge_attr_matrix, G3_edge_index, G3_edge_attr):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, edge_index, edge_attr = data.x.to(device), G3_edge_index, G3_edge_attr
        for i, layer in enumerate(self.layers):
            # In each layer's loop, the input node features x undergo graph convolution operations through the GATv2Conv layer.
            x = layer(x, edge_index, edge_weight=edge_attr)

            # Check if the current layer is the last layer.
            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        # The output x of the model passes through the log_softmax function to obtain the log probability distribution of the classes. Usually used for multi - class classification problems.
        return torch.nn.functional.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 a: float = 0.1,
                 k: int = 16,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GAT, self).__init__()

        self.k = k
        self.a = a
        # Calculate the dimension of the input features, the dimensions of the hidden layers, and the number of output classes, and store them in the num_features list.
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        # The function of zip(num_features[:-1], num_features[1:]) is to pair adjacent elements in the num_features list together.
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GATv2Conv(in_features, out_features, edge_dim=1))
        # Wrap the layers list in PyTorch's ModuleList to include them as part of the model.
        self.layers = ModuleList(layers)

        # Create a list self.reg_params that contains the parameters of the first GATv2Conv layer. These parameters may be used for regularization.
        self.reg_params = list(layers[0].parameters())
        # Create a list self.non_reg_params that contains the parameters of all GATv2Conv layers except the first one. These parameters may not be used for regularization.
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self): #A custom method to reset the parameters for each layer in the model
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data, G2_edge_attr,G1_edge_attr_matrix,G3_edge_index, G3_edge_attr):

        x, edge_index, edge_attr = data.x, G3_edge_index, G3_edge_attr
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr)

            if i == len(self.layers) - 1:#Check whether the current layer is the last layer
                break

            x = self.act_fn(x)
            x = self.dropout(x)
        # The output x of the model is passed through the log_softmax function to obtain the log-probability distribution of the class. It is often used for multi-class classification problems.
        return torch.nn.functional.log_softmax(x, dim=1)

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


class GraphSAGE_NET(torch.nn.Module):

    def __init__(self, dataset: InMemoryDataset,  hidden, dropout: float = 0.5):
        super(GraphSAGE_NET, self).__init__()
        feature=dataset.data.x.shape[1]
        classes=dataset.num_classes
        layers = []
        layers.append(SAGEConv(feature, hidden))# Define two layers of GraphSAGE
        layers.append(SAGEConv(hidden, classes))
        self.layers = ModuleList(layers)
        self.dropout = Dropout(p=dropout)

    def forward(self, data: Data, G2_edge_attr,G1_edge_attr_matrix,G3_edge_index, G3_edge_attr):
        x, edge_index= data.x, G3_edge_index

        x = self.layers[0](x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layers[1](x, edge_index)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self): #A custom method to reset the parameters for each layer in the model
        for layer in self.layers:
            layer.reset_parameters()