from networkx import convert
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

def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--projectName', help='project name')
    # parser.add_argument('--fold', help='which fold is this')
    # parser.add_argument('--datasetType', help='training or testing or testing_selected')
    parser.add_argument('--interactionDatasetName', default='NPInter2', help='raw interactions dataset')
    parser.add_argument('--createBalanceDataset', default=1, type=int, help='have you create a balance dataset when you run generate_edgelist.py, 0 means no, 1 means yes')
    parser.add_argument('--path_set_negativeInteractionKey')
    parser.add_argument('--inMemory',default=1, type=int, help='1 or 0: in memory dataset or not')
    parser.add_argument('--number', type=int)
    # parser.add_argument('--hopNumber', default=2, type=int, help='hop number of subgraph')
    parser.add_argument('--shuffle', default=1, type=int, help='shuffle interactions before generate dataset')
    parser.add_argument('--noKmer', default=0, type=int, help='Not using k-mer')
    parser.add_argument('--randomNodeEmbedding', default=0, type=int, help='1: use rangdom vector as node Embedding, 0: use node2vec')
    parser.add_argument('--output', default=1, type=int, help='output dataset or not')

    return parser.parse_args()


def return_node_list_and_edge_list():
    global interaction_list, negative_interaction_list, lncRNA_list, protein_list

    node_list = lncRNA_list[:]
    node_list.extend(protein_list)
    edge_list = interaction_list[:]
    edge_list.extend(negative_interaction_list)
    return node_list, edge_list


def read_node2vec_result(path):
    print('read node2vec result')
    node_list, edge_list = return_node_list_and_edge_list()
    serialNumber_listIndex_dict = nodeSerialNumber_listIndex_dict_generation(node_list)

    node2vec_result_file = open(path, mode='r')
    lines = node2vec_result_file.readlines()
    lines.pop(0)    # 第一行包含：节点个数 节点嵌入后的维度
    for line in lines:
        arr = line.strip().split(' ')
        serial_number = int(arr[0])
        arr.pop(0)
        node_list[serialNumber_listIndex_dict[serial_number]].embedded_vector = arr
    
    count_node_without_node2vecResult = 0
    for node in node_list:
        if len(node.embedded_vector) != 64:
            count_node_without_node2vecResult += 1
            node.embedded_vector = [0] * 64
    print(f'没有node2vec结果的节点数：{count_node_without_node2vecResult}')
    node2vec_result_file.close()


def read_random_node_embedding():
    print('use random vector as node embedding')
    global lncRNA_list, protein_list
    for lncRNA in lncRNA_list:
        lncRNA.embedded_vector = list(np.random.randn(64))
    for protein in protein_list:
        protein.embedded_vector = list(np.random.randn(64))


def load_node_k_mer(node_list, node_type, k_mer_path):
    node_name_index_dict = nodeName_listIndex_dict_generation(node_list)   # 节点的名字：节点在其所在的列表中的index
    with open(k_mer_path, mode='r') as f:   # 打开存有k-mer特征的文件
        lines = f.readlines()
        # 读取k-mer文件
        for i in range(len(lines)):
            line = lines[i]
            # 从文件中定位出lncRNA或者protein的名字
            if line[0] == '>':
                node_name = line.strip()[1:]
                if node_name in node_name_index_dict:   # 根据名字在node_list中找到它，把k-mer数据赋予它
                    node = node_list[node_name_index_dict[node_name]]
                    if len(node.attributes_vector) == 0:    # 如果一个node的attributes_vector已经被赋值过，不重复赋值
                    # 如果这个node，已经被赋予过k-mer数据，报出异常
                        if len(node.attributes_vector) != 0:
                            print(node_name, node.node_type)
                            raise Exception('node already have k-mer result')
                        # k-mer数据提取出来，根据node是lncRNA还是protein，给attributes_vector赋值
                        k_mer_vector = lines[i + 1].strip().split('\t')
                        if node_type == 'lncRNA':
                            if len(k_mer_vector) != 64:
                                raise Exception('lncRNA 3-mer error')
                            for number in k_mer_vector:
                                node.attributes_vector.append(float(number))
                            for i in range(49):
                                node.attributes_vector.append(0)
                        if node_type == 'protein':
                            if len(k_mer_vector) != 49:
                                raise Exception('protein 2-mer error')
                            for i in range(64):
                                node.attributes_vector.append(0)
                            for number in k_mer_vector:
                                node.attributes_vector.append(float(number))


def load_exam(noKmer:int, lncRNA_list:list, protein_list:list):
    if noKmer == 0:
        for lncRNA in lncRNA_list:
            if len(lncRNA.attributes_vector) != 113:
                print(len(lncRNA.attributes_vector), lncRNA.name)
                raise Exception('lncRNA.attributes_vector error')
        for protein in protein_list:
            if len(protein.attributes_vector) != 113:
                print(len(protein.attributes_vector), protein.name)
                raise Exception('protein.attributes_vector error')
    
    for lncRNA in lncRNA_list:
        if len(lncRNA.embedded_vector) != 64:
            raise Exception('lncRNA embedded_vector error')
    for protein in protein_list:
        if len(protein.embedded_vector) != 64:
            raise Exception('protein embedded_vector error')


def load_set_interactionSerialNumberPair(path) -> set:
    set_interactionSerialNumberPair = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace('\n', '')
            line = line.replace('(', '')
            line = line.replace(')', '')
            line = line.replace(' ', '')
            arr = line.split(',')
            set_interactionSerialNumberPair.add((int(arr[0]), int(arr[1])))
    return set_interactionSerialNumberPair


def load_intermediate_products(path):
    interaction_list_path = path + f'/interaction_list.txt'
    negative_interaction_list_path = path + f'/negative_interaction_list.txt'
    lncRNA_list_path = path + f'/lncRNA_list.txt'
    protein_list_path = path + f'/protein_list.txt'
    with open(file=interaction_list_path, mode='rb') as f:
        interaction_list = pickle.load(f)
    with open(file=negative_interaction_list_path, mode='rb') as f:
        negative_interaction_list = pickle.load(f)
    with open(file=lncRNA_list_path, mode='rb') as f:
        lncRNA_list = pickle.load(f)
    with open(file=protein_list_path, mode='rb') as f:
        protein_list = pickle.load(f)
    # 重新建立node和interaction的相互包含的关系
    return reset_basic_data(interaction_list, negative_interaction_list, lncRNA_list, protein_list)


def exam_list_all_interaction(list_all_interaction):
    set_key = set()
    num_pos = 0
    num_neg = 0
    for interaction in list_all_interaction:
        key = (interaction.lncRNA.serial_number, interaction.protein.serial_number)
        set_key.add(key)
        if interaction.y == 1:
            num_pos += 1
        elif interaction.y == 0:
            num_neg += 1
        else:
            raise Exception('interaction.y != 1 and interaction.y != 0')
    print(f'number of different interaction: {len(set_key)}, num of positive: {num_pos}, num of negative: {num_neg}')


def read_set_interactionKey(path):
    set_interactionKey = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split(',')
            set_interactionKey.add((int(arr[0]), int(arr[1])))
    return set_interactionKey


def build_dict_serialNumber_node(list_node):
    dict_serialNumber_node = {}
    for node in list_node:
        dict_serialNumber_node[node.serial_number] = node
    return dict_serialNumber_node


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


def exam_set_allInteractionKey_train_test(set_interactionKey_train, set_negativeInteractionKey_train, set_interactionKey_test, set_negativeInteractionKey_test):
    if len(set_interactionKey_train & set_interactionKey_test & set_negativeInteractionKey_train & set_negativeInteractionKey_test) != 0:
        raise Exception('训练集和测试集有重合')


if __name__ == "__main__":
    args = parse_args()

    # 用中间产物，重现相互作用数据集
    # path_intermediate_products_whole = f'data/intermediate_products/{args.projectName}'
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
    set_interactionKey_mutual = read_set_interactionKey(path_set_interactionKey_mutual + f'_{args.number}')
    print(set_interactionKey_mutual)

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

    # # 平衡数据集
    # while len(set_negativeInteractionKey) > len(set_interactionKey):
    #     set_negativeInteractionKey.pop()

    # load node2vec result
    node2vec_result_path = f'data/node2vec_result/{args.projectName}/result.emb'
    if args.randomNodeEmbedding == 0:
        read_node2vec_result(path=node2vec_result_path)
    else:
        read_random_node_embedding()


    # load k-mer
    if args.noKmer == 0:
        lncRNA_3_mer_path = f'data/lncRNA_3_mer/{args.interactionDatasetName}/lncRNA_3_mer.txt'
        protein_2_mer_path = f'data/protein_2_mer/{args.interactionDatasetName}/protein_2_mer.txt'
        load_node_k_mer(lncRNA_list, 'lncRNA', lncRNA_3_mer_path)
        load_node_k_mer(protein_list, 'protein', protein_2_mer_path)

    # 执行检查
    load_exam(args.noKmer, lncRNA_list, protein_list)
    
    dict_serialNumber_name_lncRNA = build_dict_serialNumber_node(lncRNA_list)
    dict_serialNumber_name_protein = build_dict_serialNumber_node(protein_list)

    # 数据集生成
    exam_list_all_interaction(interaction_list)
    exam_list_all_interaction(negative_interaction_list)
    all_interaction_list = interaction_list.copy()
    all_interaction_list.extend(negative_interaction_list)
    exam_list_all_interaction(all_interaction_list)

    if args.shuffle == 1:    # 随机打乱
        print('shuffle dataset\n')
        random.shuffle(all_interaction_list)
    
    # num_of_subgraph = len(all_interaction_list)

    if args.output == 1:
        if args.inMemory == 0:
            raise Exception('not ready')
        else:
            # 生成局部子图，不能有测试集的边
            set_interactionKey_cannotUse = set()
            set_interactionKey_cannotUse.update(set_interactionKey_mutual)

            # 生成训练集
            dataset_train_path = f'data/dataset/{args.projectName}_inMemory_train_mutualInteractionStudy_{args.number}'
            if not osp.exists(dataset_train_path):
                print(f'创建了文件夹：{dataset_train_path}')
                os.makedirs(dataset_train_path)
            set_interactionKey_forGenerate = set()
            set_interactionKey_forGenerate.update(set_interactionKey)
            set_interactionKey_forGenerate.update(set_negativeInteractionKey)
            My_trainingDataset = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(dataset_train_path, all_interaction_list, 1, set_interactionKey_forGenerate, set_interactionKey_cannotUse)

            # # 生成测试集
            dataset_test_path = f'data/dataset/{args.projectName}_inMemory_test_mutualInteractionStudy_{args.number}'
            if not osp.exists(dataset_test_path):
                print(f'创建了文件夹：{dataset_test_path}')
                os.makedirs(dataset_test_path)
            set_interactionKey_forGenerate = set()
            set_interactionKey_forGenerate.update(set_interactionKey_mutual)
            My_testingDataset = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(dataset_test_path, all_interaction_list, 1, set_interactionKey_forGenerate, set_interactionKey_cannotUse)

    exit(0)
