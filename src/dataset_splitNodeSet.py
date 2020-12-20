from torch_geometric.nn import GCNConv, TopKPooling, SAGEConv, EdgePooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch_geometric.data import Data
from torch_geometric.data import Dataset, InMemoryDataset


import torch
import pickle
from tqdm import tqdm
import networkx as nx
import gc
import os.path as osp
import os
from random import shuffle

import torch.nn.functional as F


class LncRNA_Protein_Interaction_dataset_1hop_1220_splitNodeSet_InMemory(InMemoryDataset):
    def __init__(self, root,interaction_list=None, h=None, dataset_type = None, set_serialNumber_node_train=None, set_serialNumber_node_test=None, set_serialNumber_node_test_alone=None, transform=None, pre_transform=None):
        self.interaction_list = interaction_list
        self.h = h
        self.sum_node = 0.0
        self.dataset_type = dataset_type
        self.set_serialNumber_node_train = set_serialNumber_node_train
        self.set_serialNumber_node_test = set_serialNumber_node_test
        self.set_serialNumber_node_test_alone = set_serialNumber_node_test_alone
        super(LncRNA_Protein_Interaction_dataset_1hop_1220_splitNodeSet_InMemory, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def process(self):
        # Read data into huge `Data` list.
        if self.interaction_list != None and self.h != None:
            # num_data = len(self.set_allInteractionKey_forGenerate)
            # print(f'the number of samples:{num_data}')
            data_list = []
            count = 0
            for interaction in self.interaction_list:
                if self.dataset_type == 'training':
                    if interaction.lncRNA.serial_number in self.set_serialNumber_node_train and interaction.protein.serial_number in self.set_serialNumber_node_train:
                        data = self.local_subgraph_generation(interaction, self.h)
                        data_list.append(data)
                        count = count + 1
                        if count % 100 == 0:
                            print(f'{count}')
                            print(f'average node number = {self.sum_node / count}')
                elif self.dataset_type == 'testing':
                    if interaction.lncRNA.serial_number in self.set_serialNumber_node_test and interaction.protein.serial_number in self.set_serialNumber_node_test:
                        data = self.local_subgraph_generation(interaction, self.h)
                        data_list.append(data)
                        count = count + 1
                        if count % 100 == 0:
                            print(f'{count}')
                            print(f'average node number = {self.sum_node / count}')
                elif self.dataset_type == 'testing_selected':
                    if interaction.lncRNA.serial_number in self.set_serialNumber_node_test and interaction.protein.serial_number in self.set_serialNumber_node_test:
                        if interaction.lncRNA.serial_number not in self.set_serialNumber_node_test_alone or interaction.protein.serial_number not in self.set_serialNumber_node_test_alone:
                            data = self.local_subgraph_generation(interaction, self.h)
                            data_list.append(data)
                            count = count + 1
                            if count % 100 == 0:
                                print(f'{count}')
                                print(f'average node number = {self.sum_node / count}')
                else:
                    raise Exception('dataset type has to be training, testing or testing_selected')

            print(f'the number of sample = {len(data_list)}')
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
    
    # 下面四个函数用来构建local subgraph
    def local_subgraph_generation(self, interaction, h):
        # 防止图中的回路导致无限循环，所以添加过的interaction，要存起来
        added_interaction_list = []  

        x = []
        edge_index = [[], []]

        # 子图中每个节点都得有自己独特的serial number和structural label
        # 这是为了构建edge_index
        subgraph_node_serial_number = 0

        nodeSerialNumber_subgraphNodeSerialNumber_dict = {}
        subgraphNodeSerialNumber_node_dict = {}

        # 要加入局部子图的边
        set_interactionSerialNumberPair_wait_to_add = set()
        set_interactionSerialNumberPair_wait_to_add.add((interaction.lncRNA.serial_number, interaction.protein.serial_number))

        # 给要加入局部子图中的点都分配好序号
        subgraph_serialNumber = 0
        nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction.lncRNA.serial_number] = subgraph_serialNumber
        subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction.lncRNA
        subgraph_serialNumber += 1
        nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction.protein.serial_number] = subgraph_serialNumber
        subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction.protein
        subgraph_serialNumber += 1

        if self.dataset_type == 'training':
            for interaction_temp in interaction.lncRNA.interaction_list:
                if interaction_temp.protein.serial_number in self.set_serialNumber_node_train:
                    set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
                    if interaction_temp.protein.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                        nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.protein.serial_number] = subgraph_serialNumber
                        subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.protein
                        subgraph_serialNumber += 1
            
            for interaction_temp in interaction.protein.interaction_list:
                if interaction_temp.lncRNA.serial_number in self.set_serialNumber_node_train:
                    set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
                    if interaction_temp.lncRNA.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                        nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.lncRNA.serial_number] = subgraph_serialNumber
                        subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.lncRNA
                        subgraph_serialNumber += 1
        elif self.dataset_type == 'testing' or self.dataset_type == 'testing_selected':
            for interaction_temp in interaction.lncRNA.interaction_list:
                if interaction_temp.protein.serial_number not in self.set_serialNumber_node_test:
                    set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
                    if interaction_temp.protein.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                        nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.protein.serial_number] = subgraph_serialNumber
                        subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.protein
                        subgraph_serialNumber += 1
            
            for interaction_temp in interaction.protein.interaction_list:
                if interaction_temp.lncRNA.serial_number not in self.set_serialNumber_node_test:
                    set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
                    if interaction_temp.lncRNA.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                        nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.lncRNA.serial_number] = subgraph_serialNumber
                        subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.lncRNA
                        subgraph_serialNumber += 1
        else:
            raise Exception('dataset type has to be training, testing or testing_selected')

        # 构造edge_list
        for interaction_serialNumber_pair in set_interactionSerialNumberPair_wait_to_add:
            node1_subgraphSerialNumber = nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_serialNumber_pair[0]]
            node2_subgraphSerialNumber = nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_serialNumber_pair[1]]
            edge_index[0].append(node1_subgraphSerialNumber)
            edge_index[1].append(node2_subgraphSerialNumber)
            edge_index[0].append(node2_subgraphSerialNumber)
            edge_index[1].append(node1_subgraphSerialNumber)
        
        # 构造x
        for i in range(len(subgraphNodeSerialNumber_node_dict.keys())):
            node_temp = subgraphNodeSerialNumber_node_dict[i]
            vector = []
            if i == 0 or i == 1:
                vector.append(0)
            else:
                vector.append(1)
            for f in node_temp.embedded_vector:
                vector.append(float(f))
            vector.extend(node_temp.attributes_vector)
            x.append(vector)


        # y记录这个interaction的真假
        if interaction.y == 1:
            y = [1]
        else:
            y = [0]

        self.sum_node += len(x)
        # 用x,y,edge_index创建出data，加入存放data的列表local_subgraph_list
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index)

        return data

