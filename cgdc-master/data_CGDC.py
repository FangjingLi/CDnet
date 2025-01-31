__author__ = "Stefan Weißenberger and Johannes Gasteiger"
__license__ = "MIT"

import copy
import csv
import math
# FFD
import os
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
import random as rand

from scipy.linalg import expm

import torch
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, AffinityPropagation, MeanShift, OPTICS
from sklearn.mixture import GaussianMixture

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork

# from Manual import Manual
from seeds import development_seed
from sklearn.metrics.pairwise import cosine_similarity
from torch_sparse import SparseTensor, coalesce
# from pyclustering.cluster.clarans import clarans

# DATA_PATH = 'D:/PostgraduateCode/data/'
DATA_PATH = '../../data'


def get_dataset(name: str, use_lcc: bool = True) -> InMemoryDataset:
    path = DATA_PATH
    # path = os.path.join(DATA_PATH, name)
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid("D:/PostgraduateCode/data/", name)
        # dataset = Planetoid("/root/autodl-fs/GraphDatasets/Homogeneous/", name)
    elif name =='Photo':
        # dataset = Amazon("D:/PostgraduateCode/data/pyg/Amazon", "photo")
        dataset = Amazon("/root/autodl-fs/GraphDatasets/Homogeneous/Amazon", "photo")
    elif name =='Computers':
        dataset = Amazon("/root/autodl-fs/GraphDatasets/Homogeneous/Amazon", "computers")
    elif name == 'CoauthorCS':
        dataset = Coauthor("/root/autodl-fs/GraphDatasets/Homogeneous/coauther", 'cs')
    elif name in ['cornell', 'texas', 'wisconsin', 'washington']:
        # dataset = WebKB("/root/autodl-fs/GraphDatasets/Homogeneous/webkb", name)
        dataset = WebKB(root='D:/PostgraduateCode/data/webkb',name=name)
    elif name =='actor':
        # dataset = Actor(root='D:/PostgraduateCode/data/pyg/actor')
        dataset = Actor("/root/autodl-fs/GraphDatasets/Homogeneous/actor")
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='/root/autodl-fs/GraphDatasets/Homogeneous/Wikipedia', name=name)
    elif name == 'Manual':
        raise Exception('Unknown dataset.')
    else:
        raise Exception('Unknown dataset.')

    if use_lcc:
        # The goal is to limit the data set to the maximum connected component to simplify or improve subsequent analysis or training tasks.
        # This is a common operation in graph data processing, especially when the graph is very large and contains multiple subgraphs, it is often more reasonable to focus on the maximum connected component
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


# Obtain the adjacency matrix
def get_adj_matrix(dataset: InMemoryDataset) -> np.ndarray:
    num_nodes = dataset.data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.data.edge_index[0], dataset.data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_adj_matrix1(dataset: InMemoryDataset) -> np.ndarray:
    num_nodes = dataset.data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.data.edge_index[0], dataset.data.edge_index[1]):
        adj_matrix[i, j] = 1.
        adj_matrix[j, i] = 1.
    return adj_matrix

# Calculate the transition matrix based on Personalized PageRank (PPR)
def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    # The new matrix A_tilde, which is the sum of the original adjacency matrix adj_matrix with the identity matrix This step adds the self-loop to the graph
    A_tilde = adj_matrix + np.eye(num_nodes)
    # Calculate the degree matrix D_tilde, where the diagonal element is the square root of the reciprocal degree of each node (-1/2 power)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde  # Calculate the standardized Laplacian matrix H, which is T_sym
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)  # Diffusion matrix

def get_k(adj_matrix,K0_mul):
    num_nodes = adj_matrix.shape[0]
    # A=adj_matrix+ np.eye(num_nodes) #Adjacency matrix
    # D = np.diag(1 / A.sum(axis=1)) #Metric matrix
    # T_rw= A @ D
    # The new matrix A_tilde, which is the sum of the original adjacency matrix adj_matrix and the identity matrix This step adds the self-loop to the graph
    A_tilde = adj_matrix + np.eye(num_nodes)
    # Calculate the degree matrix D_tilde, where the diagonal element is the square root of the reciprocal degree of each node (-1/2 power)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    T = D_tilde @ A_tilde @ D_tilde  # Calculate the standardized Laplacian matrix H, which is T_sym
    k=1
    T_k=np.linalg.matrix_power(T, k)
    NNZ_k =np.count_nonzero(T_k)
    while(k):
        k+=1
        NNZ_ktemp=NNZ_k
        T_k = np.linalg.matrix_power(T, k)
        NNZ_k = np.count_nonzero(T_k)
        r_k=(NNZ_k-NNZ_ktemp)/NNZ_ktemp
        if r_k==0:
            break
    K0=k
    e=math.ceil(K0/K0_mul)
    #e = math.ceil(K0 + 7)
    return e






def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))


def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    # Set the value of the adjacency matrix not in the first k indexes to 0. The purpose of this step is to remove the edges that are not retained and set their weight to 0
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    # normalization
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A / norm


def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A / norm


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
        # num_per_class: int = 2) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    if num_development>1:
        print("Homogeneous dataset split")
        development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
        test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

        train_idx = []
        rnd_state = np.random.RandomState(seed)
        for c in range(data.y.max() + 1):
            class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

        val_idx = [i for i in development_idx if i not in train_idx]
    else:
        print("Heterogeneous graph dataset split")
        development_idx = np.array([i for i in np.arange(num_nodes)])
        train_idx = []
        val_idx=[]
        rnd_state = np.random.RandomState(seed)
        for c in range(data.y.max() + 1):
            class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, int(len(class_idx)*num_development), replace=False)) #60% training per class
            val_test_idx = [i for i in class_idx if i not in train_idx]
            val_idx.extend(rnd_state.choice(val_test_idx, int(len(val_test_idx) * 0.5), replace=False))  # 20% training for each class
        test_idx = [i for i in development_idx if i not in train_idx and i not in val_idx]#Remaining test set
    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data

def analyze_weak(x, edge_index):
    x_np = x.numpy()  # Convert Tensor to a NumPy array
    # content_df = pd.DataFrame(x_np, columns=['feature_' + str(i) for i in range(x_np.shape[1])])  # Create a DataFrame
    content_df = pd.DataFrame(x.numpy(), index=np.arange(x.shape[0]),
                              columns=['feature_' + str(i) for i in range(x.shape[1])])
    content_df.index.name = 'paper_id'  # Rename the row index column to 'paper_id'

    edge_index_np = edge_index.numpy()  # Convert Tensor to a NumPy array
    cites_df = pd.DataFrame({'cited_paper_id': edge_index_np[0], 'citing_paper_id': edge_index_np[1]})  # Create a DataFrame

    # Calculate total nodes
    total_nodes = len(content_df)

    # Calculate degrees
    in_degree = cites_df.groupby('cited_paper_id').size()
    print(in_degree)
    out_degree = cites_df.groupby('citing_paper_id').size()
    total_degree = in_degree.add(out_degree, fill_value=0)
    print(total_degree)

    # Count nodes with degree <= 1
    low_degree_nodes_count = total_degree[total_degree <= 2].count()

    # Calculate proportion
    low_degree_proportion = low_degree_nodes_count / total_nodes

    # Prepare degree information for merge
    total_degree_df = total_degree.to_frame('degree').reset_index()
    total_degree_df.columns = ['paper_id', 'degree']

    # Merge total degree with content data to include node ID and class label
    degree_with_label = content_df.merge(total_degree_df, on='paper_id', how='left')
    degree_with_label['degree'] = degree_with_label['degree'].fillna(0)

    # Filter nodes with degree <= 1
    low_degree_nodes = degree_with_label[degree_with_label['degree'] <= 2]

    # Output results
    # print(f"Total number of nodes: {total_nodes}")
    # print(f"Proportion of low-degree nodes: {low_degree_proportion:.2%}")
    # print(f"Number of low-degree nodes: {len(low_degree_nodes)}")
    return low_degree_nodes.index


def delete_files_without_hamming(directory):
    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            if "hamming" not in file_name and "cos" not in file_name:
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)
        print(f"Files in folder {directory} that do not contain 'hamming' or 'cos' have been successfully deleted")
    else:
        print(f"Folder {directory} does not exist")


class PPRDataset(InMemoryDataset):
    """
    Dataset preprocessed with GDC using PPR diffusion.
    Note that this implementations is not scalable
    since we directly invert the adjacency matrix.
    """

    def __init__(self,
                 name: str = 'Cora',
                 use_lcc: bool = True,
                 alpha: float = 0.1,
                 thod: float = 0.05,
                 k: int = 16,
                 eps: float = None,
                 K0_mul: float = 1.0):
        self.thod=thod
        self.name = name
        self.use_lcc = use_lcc
        self.alpha = alpha
        self.k = k
        self.eps = eps
        self.K0_mul=K0_mul

        super(PPRDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list:
        return []

    @property
    def processed_file_names(self) -> list:
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):
        base = get_dataset(name=self.name, use_lcc=self.use_lcc)
        # generate adjacency matrix from sparse representation
        adj_matrix = get_adj_matrix(base)

        # if self.name in ['cornell','wisconsin','texas']:
        #     adj_matrix1=get_adj_matrix1(base)
        # else:
        #     adj_matrix1 = get_adj_matrix(base)

        # obtain exact PPR matrix
        ppr_matrix = get_ppr_matrix(adj_matrix,
                                    alpha=self.alpha)
        self.k=get_k(adj_matrix,self.K0_mul)
        # temp_matrix=ppr_matrix

        if self.k:
            print(f'Selecting top {self.k} edges per node.')
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=self.k)
        elif self.eps:
            print(f'Selecting edges with weight greater than {self.eps}.')
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=self.eps)
        else:
            raise ValueError

        weak_nodes = analyze_weak(base.data.x,base.data.edge_index)
        num_nodes = adj_matrix.shape[0]
        weak_mask = torch.zeros(num_nodes, dtype=torch.bool)
        weak_mask[weak_nodes] = True
        base.data.weak_mask=weak_mask

        all_data={
                  'G1_edge_attr': ppr_matrix,
                  'base':base,
                }

        data, slices = self.collate([all_data])
        torch.save((data, slices), self.processed_paths[0])

    def __str__(self) -> str:
        return f'{self.name}_FFDppr_K={self.k}_alpha={self.alpha}_eps={self.eps}_lcc={self.use_lcc}'


class HeatDataset(InMemoryDataset):
    """
    Dataset preprocessed with GDC using heat kernel diffusion.
    Note that this implementations is not scalable
    since we directly calculate the matrix exponential
    of the adjacency matrix.
    """

    def __init__(self,
                 name: str = 'Cora',
                 use_lcc: bool = True,
                 t: float = 5.0,
                 k: int = 16,
                 eps: float = None):
        self.name = name
        self.use_lcc = use_lcc
        self.t = t
        self.k = k
        self.eps = eps

        super(HeatDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list:
        return []

    @property
    def processed_file_names(self) -> list:
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):
        base = get_dataset(name=self.name, use_lcc=self.use_lcc)
        # generate adjacency matrix from sparse representation
        adj_matrix = get_adj_matrix(base)
        # get heat matrix as described in Berberidis et al., 2019
        heat_matrix = get_heat_matrix(adj_matrix,
                                      t=self.t)
        if self.k:
            print(f'Selecting top {self.k} edges per node.')
            heat_matrix = get_top_k_matrix(heat_matrix, k=self.k)
        elif self.eps:
            print(f'Selecting edges with weight greater than {self.eps}.')
            heat_matrix = get_clipped_matrix(heat_matrix, eps=self.eps)
        else:
            raise ValueError

        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(heat_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(heat_matrix[i, j])
        edge_index = [edges_i, edges_j]

        data = Data(
            x=base.data.x,
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=base.data.y,
            train_mask=torch.zeros(base.data.train_mask.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(base.data.test_mask.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(base.data.val_mask.size()[0], dtype=torch.bool)
        )

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __str__(self) -> str:
        return f'{self.name}_heat_t={self.t}_k={self.k}_eps={self.eps}_lcc={self.use_lcc}'

class HammingDistanceDataset(InMemoryDataset):
    def __init__(self,G, dataset_name: str, use_lcc: bool, alpha: float, k: int, eps: float):
        self.G = G
        self.dataset_name = dataset_name
        self.use_lcc = use_lcc
        self.alpha = alpha
        self.k = k
        self.eps = eps
        super(HammingDistanceDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):
        print("get diffusion results")
        num_nodes = self.G.data.num_nodes

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        G2_edge_attr = torch.zeros(num_nodes, num_nodes).to(device)

        print("now calculate hamming distance")
        # The hamming distance of the features of each two nodes is calculated, and the weights of the edges between the two nodes are obtained to form a complete graph G2
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # The hamming distance between node feature vectors is calculated as the weight

                hamming_dist = hamming_distance(self.G.data.x[i].int(), self.G.data.x[j].int())  # hamming calculates the proportion of different eigenvalues
                similarity = 1 - hamming_dist
                if similarity < 0.2:
                    similarity = 0
                # Fill weights into the corresponding positions in the weight matrix (undirected graphs are symmetric matrices)
                G2_edge_attr[i][j] = similarity
                G2_edge_attr[j][i] = similarity

        processed_data={
                        'G2_edge_attr':G2_edge_attr,
                        }

        data, slices = self.collate([processed_data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.dataset_name}_hamming_alpha={self.alpha}_k={self.k}_eps={self.eps}_lcc={self.use_lcc}'


class CosSimilarityDataset(InMemoryDataset):
    def __init__(self,G, dataset_name: str, use_lcc: bool, alpha: float, k: int, eps: float):
        self.G = G
        self.dataset_name = dataset_name
        self.use_lcc = use_lcc
        self.alpha = alpha
        self.k = k
        self.eps = eps
        super(CosSimilarityDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):

        print("calculate cos_similarity")

        cos_sim_matrix= cosine_similarity(self.G.data.x)
        data, slices = self.collate([cos_sim_matrix])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.dataset_name}_cos_alpha={self.alpha}_k={self.k}_eps={self.eps}_lcc={self.use_lcc}'

    

class G3Dataset:
    def __init__(self, dataset_name: str, a:float, k:int, G1_edge_attr_matrix,G2_edge_attr):
        self.dataset_name = dataset_name
        self.a=a
        self.k = k
        self.G2_edge_attr=G2_edge_attr
        self.G1_edge_attr_matrix=G1_edge_attr_matrix

    def download(self):
        pass

    def process(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("now calculate G3")
        # G1是扩散特征矩阵，G2是海明特征矩阵
        G3_edge_attr_matrix = self.a * self.G1_edge_attr_matrix + (1 - self.a) * self.G2_edge_attr  # 加权

        print(f'Selecting top {self.k} edges per node.')
        G3_edge_attr_matrix1 = get_top_k_matrix(G3_edge_attr_matrix, k=self.k)
        G3_edges_i = []
        G3_edges_j = []
        G3_edge_attr = []
        for i, row in enumerate(G3_edge_attr_matrix1):  # The outer loop traverses the rows of the matrix, where i is the index of the row and row is the contents of the row
            # The inner loop loops over elements greater than 0 in the current row, which represent pairs of nodes with links (edges).
            for j in np.where(row > 0)[0]:
                # np.where(row > 0) returns a Boolean mask indicating which elements are greater than 0, and [0] is then used to extract the index j of those elements.
                G3_edges_i.append(i)
                G3_edges_j.append(j)
                G3_edge_attr.append(G3_edge_attr_matrix1[i, j])
        edge_index = [G3_edges_i, G3_edges_j]
        G3_edge_index = torch.LongTensor(edge_index).to(device)
        G3_edge_attr = torch.FloatTensor(G3_edge_attr).to(device)
        G3 = {'G3_edge_index': G3_edge_index,
              'G3_edge_attr': G3_edge_attr
              }
        return G3

def get_edge_attr_matrix(edge_index, num_nodes, edge_attr):
    edge_attr_matrix = np.zeros(shape=(num_nodes, num_nodes))

    for i, (u, v) in enumerate(edge_index.T):
        edge_attr_matrix[u][v] = edge_attr[i]
        edge_attr_matrix[v][u] = edge_attr[i]  # Undirected graph, symmetric

    return torch.tensor(edge_attr_matrix)

def hamming_distance(tensor1, tensor2):
    xor_result = torch.bitwise_xor(tensor1, tensor2)
    hamming_dist = torch.sum(xor_result).item() / len(tensor2)
    return hamming_dist





class SNGNNWebKB(WebKB):
    def __init__(self, root: str, name: str):
        super().__init__(root, name)

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]

        train_mask = torch.stack(train_masks, dim=0)
        val_mask = torch.stack(val_masks, dim=0)
        test_mask = torch.stack(test_masks, dim=0)

        # train_mask = torch.stack(train_masks, dim=1)
        # val_mask = torch.stack(val_masks, dim=1)
        # test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])





