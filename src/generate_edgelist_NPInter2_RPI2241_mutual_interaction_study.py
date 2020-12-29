from xml.etree.ElementTree import parse
from networkx import convert
from networkx.generators.trees import prefix_tree
from networkx.readwrite.gexf import generate_gexf
from openpyxl import load_workbook
import random
import networkx as nx
import pickle
import sys
import os.path as osp
import os
import argparse
import copy
import gc
import numpy as np

from openpyxl.descriptors.base import Set
from torch import pinverse

sys.path.append(os.path.realpath('.'))

from src.classes import LncRNA
from src.classes import Protein, LncRNA_Protein_Interaction_dataset_1hop_1218, LncRNA_Protein_Interaction_dataset_1hop_1218_InMemory
from src.classes import LncRNA_Protein_Interaction, LncRNA_Protein_Interaction_dataset, LncRNA_Protein_Interaction_inMemoryDataset
from src.classes import LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory

from src.generate_edgelist import read_interaction_dataset
from src.methods import nodeSerialNumber_listIndex_dict_generation, nodeName_listIndex_dict_generation
from src.methods import reset_basic_data


from src.generate_dataset import read_set_interactionKey, build_dict_serialNumber_node

def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--projectName', help='project name')
    parser.add_argument('--interactionDatasetName', default='NPInter2', help='raw interactions dataset')
    parser.add_argument('--createBalanceDataset', default=1, type=int, help='have you create a balance dataset when you run generate_edgelist.py, 0 means no, 1 means yes')
    parser.add_argument('--path_set_negativeInteractionKey')
    parser.add_argument('--output', default=1, type=int, help='output dataset or not')
    return parser.parse_args()


def rebuild_all_negativeInteraction(set_negativeInteractionKey):
    global lncRNA_list, protein_list, negative_interaction_list
    dict_serialNumber_lncRNA = build_dict_serialNumber_node(lncRNA_list)
    dict_serialNumber_protein = build_dict_serialNumber_node(protein_list)
    # 根据set_negativeInteractionKey把负样本集构造出来
    for negativeInteractionKey in set_negativeInteractionKey:
        lncRNA_temp = dict_serialNumber_lncRNA[negativeInteractionKey[0]]
        protein_temp = dict_serialNumber_protein[negativeInteractionKey[1]]
        # 构造负样本
        temp_negativeInteraction = LncRNA_Protein_Interaction(lncRNA_temp, protein_temp, 0, negativeInteractionKey)
        negative_interaction_list.append(temp_negativeInteraction)
        lncRNA_temp.interaction_list.append(temp_negativeInteraction)
        protein_temp.interaction_list.append(temp_negativeInteraction)


def generate_G(set_interactionKey, set_negativeInteractionKey):
    G = nx.Graph()
    for interactionKey in set_interactionKey:
        G.add_edge(*interactionKey)
    for negativeInteractionKey in set_negativeInteractionKey:
        G.add_edge(*negativeInteractionKey)
    print(f'G: 节点数 = {G.number_of_nodes()}, 边数 = {G.number_of_edges()}， 连通分量数 = {len(list(nx.connected_components(G)))}')
    return G


if __name__ == "__main__":
    args = parse_args()
    interaction_dataset_path = 'data/source_database_data/'+ args.interactionDatasetName + '.xlsx'
    interaction_list, negative_interaction_list,lncRNA_list, protein_list, lncRNA_name_index_dict, protein_name_index_dict, set_interactionKey, \
        set_negativeInteractionKey = read_interaction_dataset(dataset_path=interaction_dataset_path, dataset_name=args.interactionDatasetName)
    
    # 读入mutual interaction的键集合
    if args.interactionDatasetName == 'NPInter2':
        path_set_interactionKey_mutual = r'data\NPInter2_RPI2241_mutual_interaction\interactionKey_NPInter2_mutual'
    elif args.interactionDatasetName == 'RPI2241':
        path_set_interactionKey_mutual = r'data\NPInter2_RPI2241_mutual_interaction\interactionKey_RPI2241_mutual'
    else:
        raise Exception('必须是RPI2241或者NPInter2其中之一')
    set_interactionKey_mutual = read_set_interactionKey(path_set_interactionKey_mutual)

    # 从set_interactionKey里面删去读入的NPInter2和RPI2241共有的相互作用
    for interactionKey in set_interactionKey_mutual:
        set_interactionKey.remove(interactionKey)
    

    # 读入负样本的键集合
    if args.createBalanceDataset == 1:
        path_set_negativeInteractionKey_all = args.path_set_negativeInteractionKey
        set_negativeInteractionKey = read_set_interactionKey(path_set_negativeInteractionKey_all)
        # # 随机减少负样本数量，构造平衡训练集
        # while len(set_negativeInteractionKey) > len(set_interactionKey):
        #     set_negativeInteractionKey.pop()
        # 把负样本重建在negative_interaction_list中
        rebuild_all_negativeInteraction(set_negativeInteractionKey)
    # else:
    #     while len(set_negativeInteractionKey) > len(set_interactionKey):
    #         set_negativeInteractionKey.pop()


    # 构造edgelist文件
    G = generate_G(set_interactionKey, set_negativeInteractionKey)
    path_edgelist = f'data/graph/{args.projectName}'
    if args.output == 1:
        # 保存edgelist文件
        if not osp.exists(path_edgelist):
            os.makedirs(path_edgelist)
            print(f'创建了文件夹：{path_edgelist}')
        nx.write_edgelist(G, path_edgelist+'/bipartite_graph.edgelist')
        print(f'保存了：{path_edgelist}/bipartite_graph.edgelist')
        # 创建用来保存node2vec结果的文件夹
        path_node2vec_result = f'data/node2vec_result/{args.projectName}'
        if not osp.exists(path_node2vec_result):
            os.makedirs(path_node2vec_result)
            print(f'创建了文件夹：{path_node2vec_result}')