import sys
import os.path as osp
import os
import random
from tqdm import tqdm
import time
import gc
import argparse

<<<<<<< HEAD
sys.path.append(os.path.realpath('.'))
from src.classes import Net_1, LncRNA_Protein_Interaction_dataset, LncRNA_Protein_Interaction_inMemoryDataset

=======
sys.path.append(r"C:\Python_prj\GNN_predict_rpi_0930")

from src.classes import Net_1, LncRNA_Protein_Interaction_dataset

>>>>>>> 08847d0... 20201018 有了in memory数据集，测试后提交
from src.methods import dataset_analysis, average_list, Accuracy_Precision_Sensitivity_Specificity_MCC

from torch_geometric.data import DataLoader

import torch
import torch.nn.functional as F
from torch.optim import *


def parse_args():
    parser = argparse.ArgumentParser(description="analyze dataset.")
    parser.add_argument('--datasetName', default='0930_NPInter2', help='the name of this object')
    parser.add_argument('--interactionDatasetName', default='NPInter2', help='raw interactions dataset')
    parser.add_argument('--hopNumber', default=2, help='hop number of subgraph')
    parser.add_argument('--node2vecWindowSize', default=5, help='node2vec window size')
    parser.add_argument('--onlyPositive', type=int,default=1, help='只统计正样本')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_path = f'data\\dataset\\{args.datasetName}'
    # 读取数据集
    dataset = LncRNA_Protein_Interaction_inMemoryDataset(root=dataset_path)

    average_node_number = 0
    average_edge_number = 0
    num_of_positive_data = 0
    num_of_negative_data = 0
    list_nodeNumber = []
    list_edgeNumber = []
    dict_nodeNumber_occurrenceNumber = {}
    dict_edgeNumber_occurrenceNumber = {}
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        if args.onlyPositive == 1 and data.y == True:
            # 统计节点出现次数
            if data.num_nodes in dict_nodeNumber_occurrenceNumber:
                dict_nodeNumber_occurrenceNumber[data.num_nodes] += 1
            else:
                dict_nodeNumber_occurrenceNumber[data.num_nodes] = 1
            list_nodeNumber.append(data.num_nodes)
            # 统计边出现次数
            if data.num_edges in dict_edgeNumber_occurrenceNumber:
                dict_edgeNumber_occurrenceNumber[data.num_edges] += 1
            else:
                dict_edgeNumber_occurrenceNumber[data.num_edges] = 1
            list_edgeNumber.append(data.num_edges)
            # 计算平均节点数和平均边数
            average_node_number = (average_node_number * i + data.num_nodes) / (i + 1)
            average_edge_number = (average_edge_number * i + data.num_edges) / (i + 1)
            # 检查节点特征维度
            if data.num_node_features != 178:
                print('节点的特征维数不等于178')
                print(f'特征维度：{data.num_node_features}')
                exit()
            # 统计正负样本数量
            if data.y[0] == 1:
                num_of_positive_data += 1
            else:
                num_of_negative_data += 1
        elif args.onlyPositive == 0:
            # 统计节点出现次数
            if data.num_nodes in dict_nodeNumber_occurrenceNumber:
                dict_nodeNumber_occurrenceNumber[data.num_nodes] += 1
            else:
                dict_nodeNumber_occurrenceNumber[data.num_nodes] = 1
            list_nodeNumber.append(data.num_nodes)
            # 统计边出现次数
            if data.num_edges in dict_edgeNumber_occurrenceNumber:
                dict_edgeNumber_occurrenceNumber[data.num_edges] += 1
            else:
                dict_edgeNumber_occurrenceNumber[data.num_edges] = 1
            list_edgeNumber.append(data.num_edges)
            # 计算平均节点数和平均边数
            average_node_number = (average_node_number * i + data.num_nodes) / (i + 1)
            average_edge_number = (average_edge_number * i + data.num_edges) / (i + 1)
            # 检查节点特征维度
            if data.num_node_features != 178:
                print('节点的特征维数不等于178')
                print(f'特征维度：{data.num_node_features}')
                exit()
            # 统计正负样本数量
            if data.y[0] == 1:
                num_of_positive_data += 1
            else:
                num_of_negative_data += 1
    print('图的平均节点数', average_node_number)
    print('图的平均边数', average_edge_number)
    print('正样本个数', num_of_positive_data)
    print('负样本个数', num_of_negative_data)

    # path_file = f'data/temp/{args.datasetName}'
    # if not osp.exists(path_file):
    #     os.makedirs(path_file)

    # file_nodeNumber_occurrenceNumber = open(path_file + '/nodeNumber.txt', mode='w')
    # file_nodeNumber_occurrenceNumber.write('nodeNumber\toccurrenceNumber\n')
    # for nodeNumber in sorted(dict_nodeNumber_occurrenceNumber):
    #     file_nodeNumber_occurrenceNumber.write(f'{nodeNumber}\t{dict_nodeNumber_occurrenceNumber[nodeNumber]}\n')
    
    # file_edgeNumber_occurrenceNumber = open(path_file + '/edgeNumber.txt', mode='w')
    # file_edgeNumber_occurrenceNumber.write('edgeNumber\toccurrenceNumber\n')
    # for edgeNumber in sorted(dict_edgeNumber_occurrenceNumber):
    #     file_edgeNumber_occurrenceNumber.write(f'{edgeNumber}\t{dict_edgeNumber_occurrenceNumber[edgeNumber]}\n')

    # file_nodeNumber = open(path_file + '/all_nodeNumber.txt', mode='w')
    # file_nodeNumber.write('nodeNumber\n')
    # for number in list_nodeNumber:
    #     file_nodeNumber.write(f'{number}\n')
    #
    # file_edgeNumber = open(path_file + '/all_edgeNumber.txt', mode='w')
    # file_edgeNumber.write('edgeNumber\n')
    # for number in list_edgeNumber:
    #     file_edgeNumber.write(f'{number}\n')
