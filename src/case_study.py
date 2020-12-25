import pickle
import sys, os
import argparse
from matplotlib.pyplot import text
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import os.path as osp
import time

sys.path.append(os.path.realpath('.'))

from src.generate_edgelist import read_interaction_dataset

from src.generate_dataset import exam_set_allInteractionKey_train_test, read_set_interactionKey

from src.classes import LncRNA_Protein_Interaction, LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory, Net_1

from src.generate_edgelist import read_interaction_dataset

from src.methods import nodeSerialNumber_listIndex_dict_generation, nodeName_listIndex_dict_generation
from src.methods import reset_basic_data

def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--caseStudyName', help='the name of this case study')
    parser.add_argument('--projectName', help='project name')
    parser.add_argument('--fold', help='which fold is this')
    parser.add_argument('--interactionDatasetName', default='NPInter2', help='raw interactions dataset')
    parser.add_argument('--createBalanceDataset', default=1, type=int, help='have you create a balance dataset when you run generate_edgelist.py, 0 means no, 1 means yes')
    parser.add_argument('--noKmer', default=0, type=int, help='1: Not using k-mer, 0: use k-mer')
    parser.add_argument('--randomNodeEmbedding', default=0, type=int, help='1: use rangdom vector as node Embedding, 0: use node2vec')
    parser.add_argument('--modelPath', help='the path of the model')
    # parser.add_argument('--output', default=1, type=int, help='output dataset or not')

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


def return_dict_serialNumber_name(list_node):
    dict_serialNumber_name = {}
    for node in list_node:
        dict_serialNumber_name[node.serial_number] = node.name
    return dict_serialNumber_name


def return_dict_interactionKey_interaction(list_interaction):
    dict_interactionKey_interaction = {}
    for interaction in list_interaction:
        dict_interactionKey_interaction[interaction.key] = interaction
    return dict_interactionKey_interaction


def predict_a_case(model, loader, device):
    model.eval()
    TP = 0
    FN = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim = 1)[1]
        for index in range(len(pred)):
            if pred[index] == 1:
                TP += 1
            else:
                FN += 1
    if TP == 1 and FN == 0:
        return True
    elif FN == 1 and TP == 0:
        return False
    else:
        raise Exception('there should be only one data')



if __name__ == "__main__":
    time_start = time.time()
    args = parse_args()

    interaction_dataset_path = 'data/source_database_data/'+ args.interactionDatasetName + '.xlsx'
    interaction_list, negative_interaction_list,lncRNA_list, protein_list, lncRNA_name_index_dict, protein_name_index_dict, set_interactionKey, \
        set_negativeInteractionKey = read_interaction_dataset(dataset_path=interaction_dataset_path, dataset_name=args.interactionDatasetName)
    
    # 读取随机生成的负样本的key集合
    path_set_allInteractionKey = f'data/set_allInteractionKey/{args.projectName}'
    path_set_negativeInteractionKey_all = path_set_allInteractionKey + '/set_negativeInteractionKey_all'
    if args.createBalanceDataset == 1:
        set_negativeInteractionKey = read_set_interactionKey(path_set_negativeInteractionKey_all)

    # 重建负样本
    if args.createBalanceDataset == 1:
        rebuild_all_negativeInteraction(set_negativeInteractionKey)
    
    # 把训练集和测试集包含的边读取出来
    path_set_interactionKey_train = path_set_allInteractionKey + f'/set_interactionKey_train_{args.fold}'
    path_set_negativeInteractionKey_train = path_set_allInteractionKey + f'/set_negativeInteractionKey_train_{args.fold}'
    path_set_interactionKey_test = path_set_allInteractionKey + f'/set_interactionKey_test_{args.fold}'
    path_set_negativeInteractionKey_test = path_set_allInteractionKey + f'/set_negativeInteractionKey_test_{args.fold}'

    set_interactionKey_train = read_set_interactionKey(path_set_interactionKey_train)
    set_negativeInteractionKey_train = read_set_interactionKey(path_set_negativeInteractionKey_train)
    set_interactionKey_test = read_set_interactionKey(path_set_interactionKey_test)
    set_negativeInteractionKey_test = read_set_interactionKey(path_set_negativeInteractionKey_test)

    # 检查一下训练集和测试集有没有重合
    exam_set_allInteractionKey_train_test(set_interactionKey_train, set_negativeInteractionKey_train, set_interactionKey_test, set_negativeInteractionKey_test)

    # load node2vec result
    node2vec_result_path = f'data/node2vec_result/{args.projectName}/training_{args.fold}/result.emb'
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

    # 载入模型
    device = torch.device('cuda')
    model = Net_1(178).to(device)
    model.load_state_dict(torch.load(args.modelPath))

    # 开始case study

    # 构建从serial number到名字的字典
    dict_serialNumber_lncRNAName = return_dict_serialNumber_name(lncRNA_list)
    dict_serialNumber_proteinName = return_dict_serialNumber_name(protein_list)
    dict_interactionKey_interaction = return_dict_interactionKey_interaction(interaction_list)

    set_interactionKey_cannotUse = set()
    set_interactionKey_cannotUse.update(set_interactionKey_test)
    set_interactionKey_cannotUse.update(set_negativeInteractionKey_test)
    
    device = torch.device('cuda')

    case_study_path = f'data/case_study/{args.caseStudyName}/datasets'
    if not osp.exists(case_study_path):
        os.makedirs(case_study_path)
        print(f'创建了文件夹：{case_study_path}')
    log_path = f'data/case_study/{args.caseStudyName}/logs'
    if not osp.exists(log_path):
        os.makedirs(log_path)
        print(f'创建了文件夹：{log_path}')

    log_predict_success = open(log_path+f'/case_predict_success.txt', 'w')
    log_predict_fail = open(log_path+f'/case_predict_fail.txt', 'w')

    for interaction_key in set_interactionKey_test:
        lncRNA_name = dict_serialNumber_lncRNAName[interaction_key[0]]
        protein_name = dict_serialNumber_proteinName[interaction_key[1]]
        interaction_name_pair = (lncRNA_name, protein_name)
        # 打印要预测的case的名字
        # print(interaction_name_pair)
        # 生成数据集
        set_interactionKey_forGenerate = set()
        set_interactionKey_forGenerate.add(interaction_key)
        path_dataset_caseStudy = case_study_path+f'/{lncRNA_name}_{protein_name}'
        dataset_caseStudy = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(path_dataset_caseStudy, interaction_list, 1, set_interactionKey_forGenerate, set_interactionKey_cannotUse)
        case_loader = DataLoader(dataset_caseStudy, batch_size=1)
        case_result = predict_a_case(model, case_loader, device)
        if case_result == True:
            log_predict_success.write(f'{lncRNA_name}\t{protein_name}\n')
        else:
            log_predict_fail.write(f'{lncRNA_name}\t{protein_name}\n')
    
    log_predict_fail.close()
    log_predict_success.close()
    time_end = time.time()
    print(f'总耗时：{time_end - time_start}')








    