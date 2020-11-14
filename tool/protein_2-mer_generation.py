'''
这里对protein做的k-mer是把20个氨基酸分7组
A   A,G,V
B   I,L,F,P
C   Y,M,T,S
D   H,N,Q,W
E   R,K
F   D,E
G   C
'''
import pickle
import numpy as np
import random
import os
import os.path as osp
import argparse
import sys

sys.path.append(os.path.realpath('.'))

import src.classes


def parse_args():
    parser = argparse.ArgumentParser(description="generate protein 2 mer")
    parser.add_argument('--input_fasta_file', help='input file')
    parser.add_argument('--output_folder', help='output folder')
    parser.add_argument('--k', default=2, help='k of k-mer')
    return parser.parse_args()


# def output_protein_name_list_and_sequence_list_VisFeature(dataset_name):
#     protein_name_list = []
#     protein_sequence_list = []
#     protein_sequence_file_path = 'data/protein_sequence/' + dataset_name +'/protein_sequence.fasta'
#     protein_sequence_file = open(file=protein_sequence_file_path, mode='r')
#     flag = 0    # flag=0代表现在处理的是第一个protein
#     sequence = ''
#     for line in protein_sequence_file.readlines():
#         if line[0] == '>':
#             if flag == 0:
#                 temp_list = line.strip().split('|')
#                 protein_name = temp_list[1]
#                 protein_name_list.append(protein_name)
#                 flag = flag + 1
#             else:
#                 temp_list = line.strip().split('|')
#                 protein_name = temp_list[1]
#                 protein_name_list.append(protein_name)
#                 protein_sequence_list.append(sequence)
#                 sequence = ''
#         else:
#             sequence = sequence + line.strip()
#     protein_sequence_list.append(sequence)  # 把最后一个蛋白质的sequence存入protein_sequence_list

#     print('蛋白质的数量', len(protein_name_list), '蛋白质序列记录的数量', len(protein_sequence_list))
#     protein_sequence_file.close()
#     return protein_name_list, protein_sequence_list


def output_protein_name_list_and_sequence_list(path):
    protein_name_list = []
    protein_sequence_list = []
    protein_sequence_file_path = path
    protein_sequence_file = open(protein_sequence_file_path, mode='r')
    flag = 0    # flag=0代表现在处理的是第一个protein
    sequence = ''
    for line in protein_sequence_file.readlines():  # 把protein_sequence文件中的，protein名字和protein sequence存起来
        if line[0] == '>':
            if flag == 0:
                protein_name = line.strip()[1:]
                protein_name_list.append(protein_name)
                flag = flag + 1
            else:
                protein_name = line.strip()[1:]
                protein_name_list.append(protein_name)
                protein_sequence_list.append(sequence)
                sequence = ''
        else:
            sequence = sequence + line.strip()
    protein_sequence_list.append(sequence)  # 把最后一个蛋白质的sequence存入protein_sequence_list

    print('蛋白质的数量', len(protein_name_list), '蛋白质序列记录的数量', len(protein_sequence_list))
    protein_sequence_file.close()
    return protein_name_list, protein_sequence_list


def change_protein_sequence_20_to_7():
    global protein_sequence_list
    for i in range(len(protein_sequence_list)):
        sequence_list = list(protein_sequence_list[i])
        for j in range(len(sequence_list)):
            if sequence_list[j] == 'A' or sequence_list[j] == 'G' or sequence_list[j] == 'V':
                sequence_list[j] = 'A'
            elif sequence_list[j] == 'I' or sequence_list[j] == 'L' or sequence_list[j] == 'F' or sequence_list[j] == 'P':
                sequence_list[j] = 'B'
            elif sequence_list[j] == 'Y' or sequence_list[j] == 'M' or sequence_list[j] == 'T' or sequence_list[j] == 'S':
                sequence_list[j] = 'C'
            elif sequence_list[j] == 'H' or sequence_list[j] == 'N' or sequence_list[j] == 'Q' or sequence_list[j] == 'W':
                sequence_list[j] = 'D'
            elif sequence_list[j] == 'R' or sequence_list[j] == 'K':
                sequence_list[j] = 'E'
            elif sequence_list[j] == 'D' or sequence_list[j] == 'E':
                sequence_list[j] = 'F'
            elif sequence_list[j] == 'C':
                sequence_list[j] = 'G'
            elif sequence_list[j] == 'X':
                temp = random.sample(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 1)[0]
                sequence_list[j] = temp
            else:
                print('蛋白质序列错误')
                raise Exception
        protein_sequence_list[i] = ''.join(sequence_list)


def k_mer_matrix_generation(k):
    if k == 1:
        return np.zeros([n])
    if k == 2:
        return np.zeros([n, n])
    if k == 3:
        return np.zeros([n, n, n])
    if k == 4:
        return np.zeros([n, n, n, n])
    if k == 5:
        return np.zeros([n, n, n, n, n])


def output_k_mer(num_of_protein, k, n):
    global protein_sequence_list
    len_of_vector = n ** k
    protein_k_mer_list = np.empty([num_of_protein, len_of_vector], dtype=float)
    char_index_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    for index in range(num_of_protein):  
        sequence = protein_sequence_list[index] # 取出一个蛋白质的序列
        char_list = list(sequence)  # 把序列转化成列表
        num_of_fragment = len(char_list) - k + 1
        k_mer_matrix = k_mer_matrix_generation(k)   # 根据k输出一个k维，每维长度为n的矩阵
        for i in range(len(char_list) - k + 1): # 从蛋白质序列中取出长度为k的片段
            fragment = char_list[i:i + k]
            index_list = [[0] for i in range(k)]
            for j in range(len(fragment)):
                index_list[j][0] = char_index_dict[fragment[j]]
            k_mer_matrix[tuple(index_list)] = k_mer_matrix[tuple(index_list)] + 1 / num_of_fragment
            protein_k_mer_list[index] = k_mer_matrix.reshape(n ** k)
    return protein_k_mer_list


def output_protein_k_mer_file(path):
    global protein_name_list, protein_k_mer_list
    protein_k_mer_file = open(path, mode='w')
    for i in range(len(protein_name_list)):
        protein_name = protein_name_list[i]
        protein_k_mer = protein_k_mer_list[i]
        protein_k_mer_file.write('>' + protein_name + '\n')
        flag = 0
        for num in protein_k_mer:
            if flag == 0:
                protein_k_mer_file.write(str(num))
                flag = flag + 1
            else:
                protein_k_mer_file.write('\t' + str(num))
        protein_k_mer_file.write('\n')
    protein_k_mer_file.close()

if __name__ == "__main__":
    print('\nstart\n')

    # dataset_name = 'RPI369'
    args = parse_args()

    k = args.k
    n = 7

    input_path = args.input_fasta_file
    [protein_name_list, protein_sequence_list] = output_protein_name_list_and_sequence_list(path=input_path)
    change_protein_sequence_20_to_7()

    # for protein_sequence in protein_sequence_list:
    #     for i in protein_sequence:
    #         if i == 'X':
    #             print('20氨基酸分7组后，仍然有X')

    protein_k_mer_list = output_k_mer(len(protein_name_list), k=2, n=7)

    num_of_protein = len(protein_name_list)

    outputDir_path = args.output_folder
    if not osp.exists(path=outputDir_path):
        os.makedirs(outputDir_path)

    output_path = osp.join(outputDir_path, 'protein_2_mer.txt') 
    output_protein_k_mer_file(output_path)

    print('\nexit\n')