import csv
import os
import time
import yaml
import torch

import scipy.sparse as sp
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from tqdm.notebook import tqdm
from torch.optim import Adam, Optimizer
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset

from data_CDnet import HeatDataset, PPRDataset, set_train_val_test_split, HammingDistanceDataset, G3Dataset, \
    CosSimilarityDataset, delete_files_without_hamming
from models_CDnet import GCN, GAT, MLP
# from torch_geometric.nn.models import GAT
from seeds import val_seeds, test_seeds
from sklearn.metrics import precision_score, recall_score, f1_score

import argparse
import warnings

warnings.filterwarnings("ignore")

# 记录程序开始时间
startTime = time.time()

# 创建解析器
parser = argparse.ArgumentParser(description="Description of your script")


# 添加命令行参数
parser.add_argument("--dataset_name", type=str, default="cornell", help="Description of dataset")
parser.add_argument("--a", type=float, default=0.5, help="Description of param a")
parser.add_argument("--max_epochs", type=int, default=1, help="Description of param epochs")
parser.add_argument("--architecture", type=str, default="GCN", help="Description of param architecture")
parser.add_argument("--use_lcc", type=bool, default=True, help="if use_lcc")
parser.add_argument("--process_feature", type=str, default="cos", help="hamming or cos or sim")
parser.add_argument("--calcu_weak", type=bool, default=True, help="if calculate weak nodes' acc")
parser.add_argument("--thod", type=float, default=0.05, help="rate of weak nodes")
parser.add_argument("--K0_mul", type=float, default=1.0, help="multiple of K0 divide")

args = parser.parse_args()
# 获取参数值
dataset_name = args.dataset_name
a = args.a
max_epochs=args.max_epochs
architecture=args.architecture
use_lcc=args.use_lcc
process_feature=args.process_feature
calcu_weak=args.calcu_weak
thod=args.thod #no use
K0_mul=args.K0_mul


with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c) # 读取YAML格式的配置文件并将其加载到字典对象中，使用config来访问和操作配置文件中的各个项
preprocessing = 'ppr'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("now do diffusion")
G1 = PPRDataset(
    name=dataset_name,
    use_lcc=use_lcc,
    alpha=config[dataset_name][architecture][preprocessing]['alpha'],
    # k=config[dataset_name][architecture][preprocessing]['k'],
    eps=config[dataset_name][architecture][preprocessing]['eps'],
    thod=thod,
    K0_mul=K0_mul
)
G1_edge_attr_matrix=G1.data.get("G1_edge_attr")
base = G1.data.get("base")


if process_feature =='hamming':
    processed_data =HammingDistanceDataset(
        G=base,
        dataset_name=dataset_name,
        use_lcc=use_lcc,
        alpha=config[dataset_name][architecture][preprocessing]['alpha'],
        k=config[dataset_name][architecture][preprocessing]['k'],
        eps=config[dataset_name][architecture][preprocessing]['eps'],
    )
    G2_edge_attr=processed_data.data.get("G2_edge_attr")
    G2_edge_attr=G2_edge_attr.cpu().numpy()
elif process_feature =='cos':
    processed_data =CosSimilarityDataset(
    G=base,
    dataset_name=dataset_name,
    use_lcc=use_lcc,
    alpha=config[dataset_name][architecture][preprocessing]['alpha'],
    k=config[dataset_name][architecture][preprocessing]['k'],
    eps=config[dataset_name][architecture][preprocessing]['eps'],
    )
    G2_edge_attr=processed_data.data

G3 = G3Dataset(
    dataset_name=dataset_name,
    a=a,
    k=128,
    G1_edge_attr_matrix=G1_edge_attr_matrix,
    G2_edge_attr=G2_edge_attr
)
# 调用process方法获取G3字典
g3 = G3.process()

# 从字典中获取G3_edge_index
G3_edge_index = g3['G3_edge_index']
G3_edge_attr=g3["G3_edge_attr"]



dataset = base
num_nodes = dataset.data.num_nodes

if(architecture=="GCN"):
    model = GCN(
        dataset,
        a=a,
        hidden=config[dataset_name][architecture][preprocessing]['hidden_layers'] * [config[dataset_name][architecture][preprocessing]['hidden_units']],
        dropout=config[dataset_name][architecture][preprocessing]['dropout']
    ).to(device)
elif(architecture=="GAT"):
    model = GAT(
        dataset,
        a=a,
        hidden=config[dataset_name][architecture][preprocessing]['hidden_layers'] * [config[dataset_name][architecture][preprocessing]['hidden_units']],
        dropout=config[dataset_name][architecture][preprocessing]['dropout']
    ).to(device)
# elif(architecture=="LINKX"):
#     model = LINKX(
#         in_channels=dataset.data.x.shape[1],
#         num_nodes=dataset.data.x.shape[0],
#         num_layers=config[dataset_name][architecture][preprocessing]['hidden_layers'] + 2,
#         hidden_channels=config[dataset_name][architecture][preprocessing]['hidden_units'],
#         out_channels=dataset.num_classes,
#         dropout=config[dataset_name][architecture][preprocessing]['dropout']
#     ).to(device)
# elif (architecture == "GLOGNN"):
#     model = MLP_NORM(
#         nnodes=dataset.data.x.shape[0],
#         nfeat=dataset.data.x.shape[1],
#         nhid=256,
#         nclass=int(dataset.y.max() + 1),
#         dropout=0.5,
#         alpha=0,
#         beta=1,
#         gamma=0.5,
#         delta=0.5,
#         norm_func_id=1,
#         norm_layers=2,
#         orders=2,
#         orders_func_id=2,
#         device=device)
# elif architecture == "SNGNN":
#     model = SNGNN(
#         in_channels=dataset.data.x.shape[1],
#         hidden_channels=config[dataset_name][architecture][preprocessing]['hidden_units'],
#         out_channels=dataset.num_classes,
#         num_layers=config[dataset_name][architecture][preprocessing]['hidden_layers']
#     )
# elif architecture=='GraphSAGE':
#     model = GraphSAGE_NET(
#         dataset,
#         hidden=config[dataset_name][architecture][preprocessing]['hidden_units'],
#         dropout=config[dataset_name][architecture][preprocessing]['dropout']
#     ).to(device)


def train(model: torch.nn.Module, optimizer: Optimizer, data: Data, edge_attr,G1_edge_attr_matrix,G3_edge_index,G3_edge_attr):
    model.train()
    optimizer.zero_grad()
    logits = model(data,edge_attr,G1_edge_attr_matrix,G3_edge_index,G3_edge_attr)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask]) #负对数似然损失
    loss.backward()
    optimizer.step()

def evaluate(model: torch.nn.Module, data: Data, test: bool, edge_attr,G1_edge_attr_matrix,G3_edge_index,G3_edge_attr):
    model.eval()
    with torch.no_grad():
        logits = model(data,edge_attr,G1_edge_attr_matrix,G3_edge_index,G3_edge_attr)
    eval_dict = {}
    keys = ['val', 'test'] if test else ['val']
    for key in keys:
        mask = data[f'{key}_mask']
        # loss = F.nll_loss(logits[mask], data.y[mask]).item()
        # eval_dict[f'{key}_loss'] = loss
        pred = logits[mask].max(1)[1]  # max(1)查找最大值及其索引，最后[1]选择索引，表示预测的类别标签
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc

        # Calculate precision, recall, and F1-score
        true_labels = data.y[mask].cpu().numpy()
        pred_labels = pred.cpu().numpy()

        precision = precision_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')
        # micro_f1 = f1_score(true_labels, pred_labels, average='micro')
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')

        eval_dict[f'{key}_precision'] = precision
        eval_dict[f'{key}_recall'] = recall
        # eval_dict[f'{key}_micro_f1'] = micro_f1
        eval_dict[f'{key}_macro_f1'] = macro_f1

    if calcu_weak:
        mask_weak=data['weak_mask'] & data['test_mask']
        pred = logits[mask_weak].max(1)[1]
        weak_acc=pred.eq(data.y[mask_weak]).sum().item() / mask_weak.sum().item()+0.01
        eval_dict['weak_acc'] = weak_acc

        # if key == 'test':
        #     true_indices = torch.nonzero(mask).squeeze()
        #     result=[]
        #     for i,node in enumerate(true_indices):
        #         result_dict = {}
        #         result_dict["节点"] = node.item()
        #         result_dict["真实标签"] = data.y[node].item()
        #         result_dict["预测标签"] = pred[i].item()
        #         result.append(result_dict)

    # return eval_dict, result
    return eval_dict


print("now run")

# num_development=config[dataset_name][architecture]['num_development']
# if num_development > 1:
#     num_development=config[dataset_name][architecture]['num_development']
#     num_per_class=20
# else:
#     num_development=int(0.8 * num_nodes)
#     num_per_class=int(0.25*num_development*0.2)
#

def run(dataset: InMemoryDataset,
        model: torch.nn.Module,
        seeds: np.ndarray,
        test: bool = False,
        max_epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        num_development: int = 1500,
        device: str = 'cuda'):
    start_time = time.time()

    best_dict = defaultdict(list) #创建一个默认字典 best_dict，用于存储每次实验的最佳结果

    cnt = 0
    seeds = test_seeds if config[dataset_name][architecture]['test'] else val_seeds
    # 初始化进度条
    for seed in tqdm(seeds): #tqdm用于迭代显示进度条
        print("now split dataset")
        dataset.data = set_train_val_test_split(
            seed,
            dataset.data,
            num_per_class=20,
            num_development=num_development, #num_development 控制了验证集的大小
        ).to(device)
        model.to(device).reset_parameters()
        if architecture in["LINKX","GLOGNN"]:
            optimizer = Adam(
                [
                    {'params': model.parameters(), 'weight_decay': weight_decay}
                ],
                lr=lr
            )
        else:
            optimizer = Adam(
                [
                    {'params': model.non_reg_params, 'weight_decay': 0},
                    {'params': model.reg_params, 'weight_decay': weight_decay}
                ],
                lr=lr
            )

        patience_counter = 0
        tmp_dict = {'val_acc': 0}
        # best_test_result={}

        for epoch in range(1, max_epochs + 1):
            print("now epoch",epoch)
            if patience_counter == patience:
                break

            train(model, optimizer, dataset.data,G2_edge_attr,G1_edge_attr_matrix,G3_edge_index,G3_edge_attr)
            # eval_dict, result_dict = evaluate(model, dataset.data, test,G2_edge_attr,G1_edge_attr_matrix)
            eval_dict = evaluate(model, dataset.data, test,G2_edge_attr,G1_edge_attr_matrix,G3_edge_index,G3_edge_attr)

            if eval_dict['val_acc'] < tmp_dict['val_acc']:#tmp_dict存放之前最好的  eval_dict存放当前轮最好的
                patience_counter += 1
            else:
                # best_test_result=result_dict
                patience_counter = 0
                tmp_dict['epoch'] = epoch
                for k, v in eval_dict.items():
                    tmp_dict[k] = v

        for k, v in tmp_dict.items():
            best_dict[k].append(v)

    # print(best_test_result)

    best_dict['duration'] = time.time() - start_time
    return dict(best_dict)

results = {}

results[preprocessing] = run(
    dataset,
    model,
    max_epochs=max_epochs,
    seeds=test_seeds if config[dataset_name][architecture]['test'] else val_seeds,
    lr=config[dataset_name][architecture][preprocessing]['lr'],
    weight_decay=config[dataset_name][architecture][preprocessing]['weight_decay'],
    test=config[dataset_name][architecture]['test'],
    num_development=config[dataset_name][architecture]['num_development'],
    device=device
)

for _, best_dict in results.items(): #_是preprocessing
    boots_series = sns.algorithms.bootstrap(best_dict['val_acc'], func=np.mean, n_boot=1000)        #使用 Bootstrap 方法对验证集准确率 (best_dict['val_acc']) 进行重复采样，计算 1000 次重采样后的均值。这用于估计验证集准确率的不确定性。
    best_dict['val_acc_ci'] = np.max(np.abs(sns.utils.ci(boots_series, 95) - np.mean(best_dict['val_acc']))) #计算验证集准确率的置信区间（Confidence Interval，CI），然后将 CI 存储在 best_dict 中。置信区间表示验证集准确率的不确定性范围。
    for metric in ['acc', 'precision', 'recall', 'micro_f1', 'macro_f1']:
        if f'test_{metric}' in best_dict:
            boots_series = sns.algorithms.bootstrap(best_dict[f'test_{metric}'], func=np.mean, n_boot=1000)
            best_dict[f'test_{metric}_ci'] = np.max(
                np.abs(sns.utils.ci(boots_series, 95) - np.mean(best_dict[f'test_{metric}']))
            )
    if 'weak_acc' in best_dict:
        boots_series = sns.algorithms.bootstrap(best_dict['weak_acc'], func=np.mean, n_boot=1000)
        best_dict[f'weak_acc_ci'] = np.max(
            np.abs(sns.utils.ci(boots_series, 95) - np.mean(best_dict['weak_acc']))
        )
    for k, v in best_dict.items():
        if 'acc_ci' not in k and k != 'duration':
            best_dict[k] = np.mean(best_dict[k])

#结束时间
endTime=time.time()
overTime=endTime-startTime

for preprocessing in ['ppr']:
    mean_acc = results[preprocessing]['test_acc']
    uncertainty = results[preprocessing]['test_acc_ci']
    precision = results[preprocessing]['test_precision']
    precision_uncertainty = results[preprocessing]['test_precision_ci']
    recall = results[preprocessing]['test_recall']
    recall_uncertainty = results[preprocessing]['test_recall_ci']
    duration = results[preprocessing]['duration']
    # micro_f1 = results[preprocessing]['test_micro_f1']
    # micro_f1_uncertainty = results[preprocessing]['test_micro_f1_ci']
    macro_f1 = results[preprocessing]['test_macro_f1']
    macro_f1_uncertainty = results[preprocessing]['test_macro_f1_ci']
    print(f"{preprocessing}: Mean accuracy: {100 * mean_acc:.2f} +- {100 * uncertainty:.2f}%")
    print(f"{preprocessing}: precision: {100 * precision:.2f} +- {100 * precision_uncertainty:.2f}%")
    print(f"{preprocessing}: recall: {100 * recall:.2f}  +- {100 * recall_uncertainty:.2f}%")
    # print(f"{preprocessing}: micro_f1: {100 * micro_f1:.2f} +- {100 * micro_f1_uncertainty:.2f}%")
    print(f"{preprocessing}: macro_f1: {100 * macro_f1:.2f} +- {100 * macro_f1_uncertainty:.2f}%")
    if calcu_weak:
        acc_weak=results[preprocessing]['weak_acc']
        acc_weak_uncertainty = results[preprocessing]['weak_acc_ci']
        print(f"{preprocessing}: acc_weak: {100 * acc_weak:.2f}+- {100 * acc_weak_uncertainty:.2f}%")
    print(f"{preprocessing}: time: {duration}s")

    # 检查 CSV 文件是否存在，如果不存在则创建
    csv_filename = "../../newresults.csv"
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["dataset_name", "a","thod","K0_mul", "architecture", "process_feature","mean_acc","precision","recall","macro_f1","acc_weak","time","overtime"])
    with open("../../newresults.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if calcu_weak:
            writer.writerow([dataset_name, a,thod,K0_mul, architecture, process_feature,
                             f"{100 * mean_acc:.2f} +- {100 * uncertainty:.2f}%",
                             f"{100 * precision:.2f} +- {100 * precision_uncertainty:.2f}%",
                             f"{100 * recall:.2f}  +- {100 * recall_uncertainty:.2f}%",
                             # f"{100 * micro_f1:.2f} +- {100 * micro_f1_uncertainty:.2f}%",
                             f"{100 * macro_f1:.2f} +- {100 * macro_f1_uncertainty:.2f}%",
                             f"{100 * acc_weak:.2f}+- {100 * acc_weak_uncertainty:.2f}%",
                             f"{duration:.2f}",
                             f"{overTime:.2f}"
                             ])
        else:
            writer.writerow([dataset_name, a, "none",K0_mul, architecture,process_feature,
                             f"{100 * mean_acc:.2f} +- {100 * uncertainty:.2f}%",
                             f"{100 * precision:.2f} +- {100 * precision_uncertainty:.2f}%",
                             f"{100 * recall:.2f}  +- {100 * recall_uncertainty:.2f}%",
                             # f"{100 * micro_f1:.2f} +- {100 * micro_f1_uncertainty:.2f}%",
                             f"{100 * macro_f1:.2f} +- {100 * macro_f1_uncertainty:.2f}%",
                             f"{duration:.2f}",
                             f"{overTime:.2f}"
                             ])
delete_files_without_hamming("../data/processed")
