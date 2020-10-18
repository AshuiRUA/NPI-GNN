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

sys.path.append(r"C:\Python_prj\GNN_predict_rpi_0930")

from src.classes import LncRNA
from src.classes import Protein
from src.classes import LncRNA_Protein_Interaction, LncRNA_Protein_Interaction_dataset, LncRNA_Protein_Interaction_inMemoryDataset

from src.methods import nodeSerialNumber_listIndex_dict_generation, nodeName_listIndex_dict_generation
from src.methods import load_intermediate_products

def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--projectName', default='0930_NPInter2', help='project name')
    parser.add_argument('--interactionDatasetName', default='NPInter2', help='raw interactions dataset')
    parser.add_argument('--inMemory',type=int, help='in memory dataset or not')
    parser.add_argument('--hopNumber', default=2, type=int, help='hop number of subgraph')
    parser.add_argument('--shuffle', default=1, type=int, help='shuffle interactions before generate dataset')
    parser.add_argument('--noKmer', default=0, type=int, help='Not using k-mer')
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
    print('读入，node2vec的结果')
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
    
    for node in node_list:
        if len(node.embedded_vector) != 64:
            print('node2vec读入有问题')
    node2vec_result_file.close()


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
                            raise Exception('node被赋予过k-mer数据')
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


if __name__ == "__main__":
    args = parse_args()

    # 用中间产物，重现相互作用数据集
    interaction_list, negative_interaction_list, lncRNA_list, protein_list = load_intermediate_products(args.projectName)

    # load node2vec result
    node2vec_result_path = f'data/node2vec_result/{args.projectName}/result.emb'
    read_node2vec_result(path=node2vec_result_path)

    # load k-mer
    if args.noKmer == 0:
        lncRNA_3_mer_path = f'data/lncRNA_3_mer/{args.interactionDatasetName}/lncRNA_3_mer.txt'
        protein_2_mer_path = f'data/protein_2_mer/{args.interactionDatasetName}/protein_2_mer.txt'
        load_node_k_mer(lncRNA_list, 'lncRNA', lncRNA_3_mer_path)
        load_node_k_mer(protein_list, 'protein', protein_2_mer_path)

    # 执行检查
    load_exam(args.noKmer, lncRNA_list, protein_list)
    
    
    # 数据集生成
    all_interaction_list = interaction_list.copy()
    all_interaction_list.extend(negative_interaction_list)
    if args.shuffle == 1:    # 随机打乱
        print('shuffle dataset\n')
        random.shuffle(all_interaction_list)
    
    num_of_subgraph = len(all_interaction_list)
    dataset_path = f'data\\dataset\\{args.projectName}'
    if not osp.exists(dataset_path):
        os.makedirs(dataset_path)
    if args.output == 1:
        if args.inMemory == 0:
            My_dataset = LncRNA_Protein_Interaction_dataset(root=dataset_path, interaction_list=all_interaction_list, h=args.hopNumber)
        else:
            My_dataset = LncRNA_Protein_Interaction_inMemoryDataset(root=dataset_path, interaction_list=all_interaction_list, h=args.hopNumber)
    print('\n' + 'exit' + '\n')