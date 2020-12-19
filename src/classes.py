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

class Node:
    def __init__(self, name, serial_number, node_type):
        self.name = name
        self.interaction_list = []
        self.serial_number = serial_number  # 从0开始的序号
        self.embedded_vector = []
        self.attributes_vector = []
        self.node_type = node_type 

class LncRNA:
    def __init__(self, lncRNA_name, serial_number, node_type):
        Node.__init__(self, lncRNA_name, serial_number, node_type)
        

class Protein:
    def __init__(self, protein_name, serial_number, node_type):
        Node.__init__(self, protein_name, serial_number, node_type)

class LncRNA_Protein_Interaction:
    def __init__(self, lncRNA, protein, y:int, key=None):
        self.lncRNA = lncRNA
        self.protein = protein
        self.y = y  #y=1代表真的连接，y=0代表假的连接
        self.key = key


class Net_1(torch.nn.Module):
    def __init__(self, num_node_features, num_of_classes=2):
        super(Net_1, self).__init__()
        self.conv1 = SAGEConv(num_node_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.5)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.5)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.5)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_of_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)

        return x


class LncRNA_Protein_Interaction_dataset(Dataset):
    
    def __init__(self, root, interaction_list=None, h=None, transform=None, pre_transform=None):
        self.processed_file_names_list = []
        self.interaction_list = interaction_list
        self.h = h
        self.root = root
        if interaction_list != None:
            self.num_of_subgraph = len(interaction_list)
            for i in range(self.num_of_subgraph):
                self.processed_file_names_list.append('data_{}.pt'.format(i))
        else:
            self.processed_file_names_list = os.listdir(self.processed_dir)
            self.processed_file_names_list.remove('pre_filter.pt')
            self.processed_file_names_list.remove('pre_transform.pt')
            self.num_of_subgraph = len(self.processed_file_names_list)
        super(LncRNA_Protein_Interaction_dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return self.processed_file_names_list

    def download(self):
        pass

    def process(self):
        for i in range(self.num_of_subgraph):
            interaction = self.interaction_list[i]
            data = self.local_subgraph_generation(interaction, self.h)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            if i%100 == 0:
                print(f'{i}/{self.num_of_subgraph}')
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    
    def len(self):
        return len(self.processed_file_names)
    
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

        # 最重要的函数，把interaction的h-hop子网中的，节点和边都加入，data.x和data.edge_index
        # add_interaction_to_data(interaction=interaction, hop_count=0, h=h)
        self.add_interaction_to_data(interaction, 0, h, x, edge_index, nodeSerialNumber_subgraphNodeSerialNumber_dict, 
                                added_interaction_list, subgraph_node_serial_number)


        # y记录这个interaction的真假
        if interaction.y == 1:
            y = [1]
        else:
            y = [0]


        # 用x,y,edge_index创建出data，加入存放data的列表local_subgraph_list
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index)

        return data
    
    def add_interaction_to_data(self, interaction, hop_count, h, x, edge_index, nodeSerialNumber_subgraphNodeSerialNumber_dict,
                                added_interaction_list, subgraph_node_serial_number):
        # h限制的子图的大小
        if hop_count > h:   
            return
        # 如果没有超出h跳子图
        
        # 用interaction两边的节点的serial number构成的元组，作为他的唯一标识
        interaction_label = (interaction.lncRNA.serial_number, interaction.protein.serial_number)
        if added_interaction_list.count(interaction_label) == 0:    # 没遍历过这个interaction
            added_interaction_list.append(interaction_label)    # 把interaction的标识，加入已遍历的列表
            node_0 = interaction.lncRNA
            node_1 = interaction.protein
            # 如果node_0还没遍历到过，需要把它的特征向量加入data.x，并且分配subgraph_node_serial_number
            if node_0.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict:  
                nodeSerialNumber_subgraphNodeSerialNumber_dict[node_0.serial_number] = subgraph_node_serial_number
                subgraph_node_serial_number += 1
                self.add_node_to_x(node=node_0, structural_label=hop_count, x=x)
            # 如果node_1还没遍历到过，需要把它的特征向量加入data.x，并且分配subgraph_node_serial_number
            if node_1.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict:
                nodeSerialNumber_subgraphNodeSerialNumber_dict[node_1.serial_number] = subgraph_node_serial_number
                subgraph_node_serial_number += 1
                self.add_node_to_x(node=node_1, structural_label=hop_count, x=x)
            # 因为这是一个没遍历过的interaction，所以要在data.edge_index中记录它
            self.add_interaction_to_edge_index(serial_number_1=nodeSerialNumber_subgraphNodeSerialNumber_dict[node_0.serial_number],
                                        serial_number_2=nodeSerialNumber_subgraphNodeSerialNumber_dict[node_1.serial_number],
                                        edge_index=edge_index)
            # 然后把node_0的其它interaction也加入子网
            next_hop_count = hop_count + 1
            for temp_interaction in node_0.interaction_list:
                self.add_interaction_to_data(temp_interaction, next_hop_count, h, x, edge_index, 
                                        nodeSerialNumber_subgraphNodeSerialNumber_dict, added_interaction_list, 
                                        subgraph_node_serial_number)
            # 然后把node_1的其它interaction也加入子网
            for temp_interaction in node_1.interaction_list:
                self.add_interaction_to_data(temp_interaction, next_hop_count, h, x, edge_index, 
                                        nodeSerialNumber_subgraphNodeSerialNumber_dict, added_interaction_list, 
                                        subgraph_node_serial_number)
        else:   # interaction已经被遍历过
            return
    
    def add_node_to_x(self, node, structural_label, x):
        vector = [structural_label]
        # embedded_vector里面都是字符串，不是数
        for i in node.embedded_vector:
            vector.append(float(i))
        vector.extend(node.attributes_vector)
        x.append(vector)

    def add_interaction_to_edge_index(self, serial_number_1, serial_number_2, edge_index):
        edge_index[0].append(serial_number_1)
        edge_index[1].append(serial_number_2)
        edge_index[0].append(serial_number_2)
        edge_index[1].append(serial_number_1)


class LncRNA_Protein_Interaction_inMemoryDataset(InMemoryDataset):
    def __init__(self, root,interaction_list=None, h=None, transform=None, pre_transform=None):
        self.interaction_list = interaction_list
        self.h = h
        super(LncRNA_Protein_Interaction_inMemoryDataset, self).__init__(root, transform, pre_transform)
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
            num_data = len(self.interaction_list)
            print(f'the number of samples:{num_data}')
            data_list = []
            count = 0
            for interaction in self.interaction_list:
                data = self.local_subgraph_generation(interaction, self.h)
                data_list.append(data)
                count = count + 1
                if count % 100 == 0:
                    print(f'{count}/{num_data}')

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

        # 最重要的函数，把interaction的h-hop子网中的，节点和边都加入，data.x和data.edge_index
        # add_interaction_to_data(interaction=interaction, hop_count=0, h=h)
        self.add_interaction_to_data(interaction, 0, h, x, edge_index, nodeSerialNumber_subgraphNodeSerialNumber_dict, 
                                added_interaction_list, subgraph_node_serial_number)


        # y记录这个interaction的真假
        if interaction.y == 1:
            y = [1]
        else:
            y = [0]


        # 用x,y,edge_index创建出data，加入存放data的列表local_subgraph_list
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index)

        return data
    
    def add_interaction_to_data(self, interaction, hop_count, h, x, edge_index, nodeSerialNumber_subgraphNodeSerialNumber_dict,
                                added_interaction_list, subgraph_node_serial_number):
        # h限制的子图的大小
        if hop_count > h:   
            return
        # 如果没有超出h跳子图
        
        # 用interaction两边的节点的serial number构成的元组，作为他的唯一标识
        interaction_label = (interaction.lncRNA.serial_number, interaction.protein.serial_number)
        if added_interaction_list.count(interaction_label) == 0:    # 没遍历过这个interaction
            added_interaction_list.append(interaction_label)    # 把interaction的标识，加入已遍历的列表
            node_0 = interaction.lncRNA
            node_1 = interaction.protein
            # 如果node_0还没遍历到过，需要把它的特征向量加入data.x，并且分配subgraph_node_serial_number
            if node_0.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict:  
                nodeSerialNumber_subgraphNodeSerialNumber_dict[node_0.serial_number] = subgraph_node_serial_number
                subgraph_node_serial_number += 1
                self.add_node_to_x(node=node_0, structural_label=hop_count, x=x)
            # 如果node_1还没遍历到过，需要把它的特征向量加入data.x，并且分配subgraph_node_serial_number
            if node_1.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict:
                nodeSerialNumber_subgraphNodeSerialNumber_dict[node_1.serial_number] = subgraph_node_serial_number
                subgraph_node_serial_number += 1
                self.add_node_to_x(node=node_1, structural_label=hop_count, x=x)
            # 因为这是一个没遍历过的interaction，所以要在data.edge_index中记录它
            self.add_interaction_to_edge_index(serial_number_1=nodeSerialNumber_subgraphNodeSerialNumber_dict[node_0.serial_number],
                                        serial_number_2=nodeSerialNumber_subgraphNodeSerialNumber_dict[node_1.serial_number],
                                        edge_index=edge_index)
            # 然后把node_0的其它interaction也加入子网
            next_hop_count = hop_count + 1
            for temp_interaction in node_0.interaction_list:
                self.add_interaction_to_data(temp_interaction, next_hop_count, h, x, edge_index, 
                                        nodeSerialNumber_subgraphNodeSerialNumber_dict, added_interaction_list, 
                                        subgraph_node_serial_number)
            # 然后把node_1的其它interaction也加入子网
            for temp_interaction in node_1.interaction_list:
                self.add_interaction_to_data(temp_interaction, next_hop_count, h, x, edge_index, 
                                        nodeSerialNumber_subgraphNodeSerialNumber_dict, added_interaction_list, 
                                        subgraph_node_serial_number)
        else:   # interaction已经被遍历过
            return
    
    def add_node_to_x(self, node, structural_label, x):
        vector = [structural_label]
        # embedded_vector里面都是字符串，不是数
        for i in node.embedded_vector:
            vector.append(float(i))
        vector.extend(node.attributes_vector)
        x.append(vector)

    def add_interaction_to_edge_index(self, serial_number_1, serial_number_2, edge_index):
        edge_index[0].append(serial_number_1)
        edge_index[1].append(serial_number_2)
        edge_index[0].append(serial_number_2)
        edge_index[1].append(serial_number_1)


class LncRNA_Protein_Interaction_dataset_1hop_1218(Dataset):
    
    def __init__(self, root, interaction_list=None, h=None, transform=None, pre_transform=None):
        self.processed_file_names_list = []
        self.interaction_list = interaction_list
        self.h = h
        self.root = root
        if interaction_list != None:
            self.num_of_subgraph = len(interaction_list)
            for i in range(self.num_of_subgraph):
                self.processed_file_names_list.append('data_{}.pt'.format(i))
        else:
            self.processed_file_names_list = os.listdir(self.processed_dir)
            self.processed_file_names_list.remove('pre_filter.pt')
            self.processed_file_names_list.remove('pre_transform.pt')
            self.num_of_subgraph = len(self.processed_file_names_list)
        super(LncRNA_Protein_Interaction_dataset_1hop_1218, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return self.processed_file_names_list

    def download(self):
        pass

    def process(self):
        for i in range(self.num_of_subgraph):
            interaction = self.interaction_list[i]
            data = self.local_subgraph_generation(interaction, self.h)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            if i%100 == 0:
                print(f'{i}/{self.num_of_subgraph}')
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    
    def len(self):
        return len(self.processed_file_names)
    
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

        for interaction_temp in interaction.lncRNA.interaction_list:
            set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
            if interaction_temp.protein.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.protein.serial_number] = subgraph_serialNumber
                subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.protein
                subgraph_serialNumber += 1
        
        for interaction_temp in interaction.protein.interaction_list:
            set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
            if interaction_temp.lncRNA.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.lncRNA.serial_number] = subgraph_serialNumber
                subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.lncRNA
                subgraph_serialNumber += 1

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


        # 用x,y,edge_index创建出data，加入存放data的列表local_subgraph_list
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index)

        return data


class LncRNA_Protein_Interaction_dataset_1hop_1218_InMemory(InMemoryDataset):
    def __init__(self, root,interaction_list=None, h=None, transform=None, pre_transform=None):
        self.interaction_list = interaction_list
        self.h = h
        self.sum_node = 0.0
        super(LncRNA_Protein_Interaction_dataset_1hop_1218_InMemory, self).__init__(root, transform, pre_transform)
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
            num_data = len(self.interaction_list)
            print(f'the number of samples:{num_data}')
            data_list = []
            count = 0
            for interaction in self.interaction_list:
                data = self.local_subgraph_generation(interaction, self.h)
                data_list.append(data)
                count = count + 1
                if count % 100 == 0:
                    print(f'{count}/{num_data}')
                    print(f'average node number = {self.sum_node / count}')

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

        for interaction_temp in interaction.lncRNA.interaction_list:
            set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
            if interaction_temp.protein.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.protein.serial_number] = subgraph_serialNumber
                subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.protein
                subgraph_serialNumber += 1
        
        for interaction_temp in interaction.protein.interaction_list:
            set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
            if interaction_temp.lncRNA.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.lncRNA.serial_number] = subgraph_serialNumber
                subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.lncRNA
                subgraph_serialNumber += 1

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


class LncRNA_Protein_Interaction_dataset_1hop_1218_dataType_InMemory(InMemoryDataset):
    def __init__(self, root, dataset_type, interaction_list=None, h=None, set_allInteractionSerialNumberPair_training = None, \
        set_allInteractionSerialNumberPair_testing=None, set_allInteractionSerialNumberPair_testing_selected = None, \
        set_allInteractionSerialNumberPair_between = None, transform=None, pre_transform=None):

        self.interaction_list = interaction_list
        self.h = h
        self.set_allInteractionSerialNumberPair_training = set_allInteractionSerialNumberPair_training
        self.set_allInteractionSerialNumberPair_testing = set_allInteractionSerialNumberPair_testing
        self.set_allInteractionSerialNumberPair_testing_selected = set_allInteractionSerialNumberPair_testing_selected
        self.set_allInteractionSerialNumberPair_between = set_allInteractionSerialNumberPair_between
        self.dataset_type = dataset_type
        self.avr_x_size = 0.0
        self.avr_positive = 0.0
        self.avr_negative = 0.0
        if interaction_list != None:
            # 决定根据谁生成数据集
            if dataset_type == 'training':
                self.set_allInteractionSerialNumberPair_toGenerate = self.set_allInteractionSerialNumberPair_training
                self.set_allInteractionSerialNumberPair_canNotUse = set()
                # self.set_allInteractionSerialNumberPair_canNotUse.update(self.set_allInteractionSerialNumberPair_testing)
                self.set_allInteractionSerialNumberPair_canNotUse.update(self.set_allInteractionSerialNumberPair_between)
            elif dataset_type == 'testing':
                self.set_allInteractionSerialNumberPair_toGenerate = self.set_allInteractionSerialNumberPair_testing
                self.set_allInteractionSerialNumberPair_canNotUse = set()
                self.set_allInteractionSerialNumberPair_canNotUse.update(self.set_allInteractionSerialNumberPair_testing)
            elif dataset_type == 'testing_selected':
                self.set_allInteractionSerialNumberPair_toGenerate = self.set_allInteractionSerialNumberPair_testing_selected
                self.set_allInteractionSerialNumberPair_canNotUse = set()
                self.set_allInteractionSerialNumberPair_canNotUse.update(self.set_allInteractionSerialNumberPair_testing)
            else:
                raise Exception("dataset_type has to be train, test or test_selected")

        super(LncRNA_Protein_Interaction_dataset_1hop_1218_dataType_InMemory, self).__init__(root, transform, pre_transform)
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
            num_data = len(self.set_allInteractionSerialNumberPair_toGenerate)
            print(f'the number of samples:{num_data}')
            data_list = []
            count = 0
            for interaction in self.interaction_list:
                # 挑出训练集中相互作用，加入数据集
                if (interaction.lncRNA.serial_number, interaction.protein.serial_number)  in self.set_allInteractionSerialNumberPair_toGenerate:
                    data = self.local_subgraph_generation(interaction, self.h)
                    data_list.append(data)
                    count = count + 1
                    if count % 100 == 0:
                        print(f'{count}/{num_data}')
                        print(f'average node number = {self.avr_x_size/count}')
                        print(f'average positive edge = {self.avr_positive / count}')
                        print(f'average negative edge = {self.avr_negative / count}')
            if len(data_list) != num_data:
                raise Exception('len(data_list) != num_data')

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

        positive_count = 0
        negative_count = 0

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

    
        for interaction_temp in interaction.lncRNA.interaction_list:
            if (interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number) not in self.set_allInteractionSerialNumberPair_canNotUse:
                if interaction_temp.y == 1:
                    positive_count += 1
                else:
                    negative_count += 1
                set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
                if interaction_temp.protein.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                    nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.protein.serial_number] = subgraph_serialNumber
                    subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.protein
                    subgraph_serialNumber += 1
        
        for interaction_temp in interaction.protein.interaction_list:
            if (interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number) not in self.set_allInteractionSerialNumberPair_canNotUse:
                if interaction_temp.y == 1:
                    positive_count += 1
                else:
                    negative_count += 1
                set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
                if interaction_temp.lncRNA.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                    nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.lncRNA.serial_number] = subgraph_serialNumber
                    subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.lncRNA
                    subgraph_serialNumber += 1

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

        self.avr_x_size += len(x)
        self.avr_positive += positive_count
        self.avr_negative += negative_count
        # 用x,y,edge_index创建出data，加入存放data的列表local_subgraph_list
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index)

        return data