from pickle import load
import sys
import argparse
from numpy.core.defchararray import index
from torch_geometric.data import DataLoader
import os.path as osp
import os
import torch
import matplotlib.pyplot as plt
import math
import gc
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

sys.path.append(os.path.realpath('.'))
from src.classes import LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory, Net_1
from src.methods import Accuracy_Precision_Sensitivity_Specificity_MCC


def AUROC(model, loader, device):
    model.eval()
    p_positive = []
    labels = []
    for data in loader:
        data = data.to(device)
        model_output = model(data)
        for val in model_output[:,1]:
            p_positive.append(math.exp(val))
        for val in data.y:
            labels.append(int(val))
    
    AUROC_result = roc_auc_score(labels, p_positive)
    fpr, tpr, _ = roc_curve(labels, p_positive)
    precision, recall, _ = precision_recall_curve(labels, p_positive)
    return AUROC_result, fpr, tpr, precision, recall



def return_scores_and_labels(model, loader, device):
    scores_loged = []
    labels_temp = []
    for data in loader :
        # data = data.to(device)
        model_output = model(data)
        scores_loged.extend(model_output[:,1])
        labels_temp.extend(data.y)
        gc.collect()
    scores = []
    labels = []
    for score_loged in scores_loged:
        scores.append(math.exp(score_loged))
    for label in labels_temp:
        labels.append(int(label))
    return scores, labels
            


def return_scores_and_labels_for_5fold(i, type):
    # 有kmer的
        # load datset
        print('load dataset with kmer')
        datset_test = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(f'data/dataset/1227_1_inMemory{type}_test_{i}')
        print(f'载入数据集：data/dataset/1227_1_inMemory{type}_test_{i}')
        test_loader = DataLoader(datset_test, batch_size=60)
        # load model
        print('load model with kmer')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net_1(datset_test.num_node_features)
        model.load_state_dict(torch.load(f'result/1227_1{type}/model_{i}_fold/10'))
        print(f'载入模型：result/1227_1{type}/model_{i}_fold/10')
        # run test
        scores_temp, labels_temp = return_scores_and_labels(model, test_loader, device)
        return scores_temp, labels_temp


def return_TP_FN_TN_FP_list(scores:list, labels:list, list_index_socore_min_to_max:list):
    list_threshold = []
    list_TP = []
    list_FN = []
    list_TN = []
    list_FP = []
    for i in range(len(list_index_socore_min_to_max)):
        # 确定阈值
        index_threshold = list_index_socore_min_to_max[i]
        threshold = scores[index_threshold]
        list_threshold.append(threshold)
        TP = FN = TN = FP = 0
        preds = [1] * len(list_index_socore_min_to_max)
        for j in range(i):
            preds[list_index_socore_min_to_max[j]] = 0

        for j in range(len(preds)):
            pred = preds[j]
            if pred == 1:
                if labels[j] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if labels[j] == 0:
                    TN += 1
                else:
                    FN += 1

        list_TP.append(TP)
        list_FN.append(FN)
        list_TN.append(TN)
        list_FP.append(FP)
    # 最后加上一种情况，就是全部预测为负
    list_TP.append(0)
    list_FN.append(labels.count(1))
    list_TN.append(labels.count(0))
    list_FP.append(0)
    return list_TP, list_FN, list_TN, list_FP


def return_list_recall_pre_spe(list_TP, list_FN, list_TN, list_FP):
    list_recall = []
    list_pre = []
    list_spe = []
    if not (len(list_TP) == len(list_FN) and len(list_FN) == len(list_TN) and len(list_TN) == len(list_FP)):
        raise Exception('长度不对等')
    for i in range(len(list_TP)):
        TP = list_TP[i]
        FN = list_FN[i]
        TN = list_TN[i]
        FP = list_FP[i]
        if TP == 0:
            list_recall.append(0.0)
            list_pre.append(0.0)
        else:
            list_recall.append(TP / (TP + FN))
            list_pre.append(TP / (TP+FP))
        if TN == 0:
            list_spe.append(0.0)
        else:
            list_spe.append(TN / (TN + FP))
    return list_recall, list_pre, list_spe



def write_list(list_value,path):
    with open(path, 'w') as f:
        for value in list_value:
            f.write(f'{value}\t')


def return_au_PR(list_pre, list_sen):
    last_sen = 1.0
    AUPR = 0
    for i in range(len(list_pre)):
        pre = list_pre[i]
        sen = list_sen[i]
        AUPR += (last_sen-sen) * (pre-0.5)
        last_sen = sen
    return AUPR
        


if __name__ == "__main__":
    # 计算有kmer的5折测试结果
    type = ''
    scores_withKmer = []
    labels_withKmer = []
    for i in range(5):
        scores_temp, labels_temp = return_scores_and_labels_for_5fold(i, type)
        scores_withKmer.extend(scores_temp)
        labels_withKmer.extend(labels_temp)
    

    AUROC_withKmer = roc_auc_score(labels_withKmer, scores_withKmer)
    

    # 计算没有kmer的5折测试结果
    type = '_noKmer'
    scores_withoutKmer = []
    labels_withoutKmer = []
    for i in range(5):
        scores_temp, labels_temp = return_scores_and_labels_for_5fold(i, type)
        scores_withoutKmer.extend(scores_temp)
        labels_withoutKmer.extend(labels_temp)
    

    AUROC_withoutKmer = roc_auc_score(labels_withoutKmer, scores_withoutKmer)
    


    #画ROC
    fpr_withKmer, tpr_withKmer, _ = roc_curve(labels_withKmer, scores_withKmer, pos_label=1)
    fpr_withoutKmer, tpr_withoutKmer, _ = roc_curve(labels_withoutKmer, scores_withoutKmer, pos_label=1)

    plt.figure(1)
    plt.plot(fpr_withKmer, tpr_withKmer, label='with kmer', color='r')
    plt.plot(fpr_withoutKmer, tpr_withoutKmer, label = 'without kmer', color='g')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    print('ROC curve')
    plt.savefig(fname = f'figure_for_paper/ROC_1227_1_10epoch.svg', format='svg')
    plt.show()
    print(f'with k-mer, AUROC = {AUROC_withKmer}')
    print(f'without k-mer, AUROC = {AUROC_withoutKmer}')
    #画PR曲线
    pre_withKmer, sen_withKmer, _ = precision_recall_curve(labels_withKmer, scores_withKmer)  
    pre_withoutKmer, sen_withoutKmer, _ = precision_recall_curve(labels_withoutKmer, scores_withoutKmer)

    plt.figure(2)
    plt.plot(sen_withKmer, pre_withKmer, label='with kmer', color='r')
    plt.plot(sen_withoutKmer, pre_withoutKmer, label = 'without kmer', color='g')
    plt.xlim((0, 1))
    plt.ylim((0.5, 1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    print('PR curve')
    plt.savefig(fname = f'figure_for_paper/PR_1227_1_10epoch.svg', format='svg')
    plt.show()
    AUPR_withKmer = return_au_PR(pre_withKmer, sen_withKmer)
    AUPR_withoutKmer = return_au_PR(pre_withoutKmer, sen_withoutKmer)
    print(f'with k-mer, AUPR = {AUPR_withKmer}')
    print(f'without k-mer, AUPR = {AUPR_withoutKmer}')
    

    # 对scores排序，同时改变label的顺序
    # list_index_socore_min_to_max = np.argsort(scores)


    # list_TP, list_FN, list_TN, list_FP = return_TP_FN_TN_FP_list(scores, labels, list_index_socore_min_to_max)
    # list_recall, list_pre, list_spe = return_list_recall_pre_spe(list_TP, list_FN, list_TN, list_FP)
    # write_list(list_recall, f'data/compare_withOrwithout_kmer/list_recall{type}')
    # write_list(list_pre, f'data/compare_withOrwithout_kmer/list_pre{type}')
    # write_list(list_spe, f'data/compare_withOrwithout_kmer/list_spe{type}')


    
    
    
    # AUROC_result_withKmer, fpr_withKmer, tpr_withKmer, precision_withKmer, recall_withKmer = AUROC(model, test_loader, device)

    # #没有kmer的
    # # load dataset
    # print('load dataset without kmer')
    # dataset_test = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(r'data\dataset\1223_1_inMemory_noKmer_test_0')
    # test_loader = DataLoader(dataset_test, batch_size=200)
    # # load model
    # print('load model without kmer')
    # device = torch.device('cuda')
    # model = Net_1(dataset_test.num_node_features).to(device)
    # model.load_state_dict(torch.load(r'result\1223_1_noKmer\model_0_fold\35'))
    # # run test
    # AUROC_result_noKmer, fpr_noKmer, tpr_noKmer, precision_noKmer, recall_noKmer = AUROC(model, test_loader, device)

    # print('有kmer')
    # print(f'AUROC = {AUROC_result_withKmer}')
    # print('没有kmer')
    # print(f'AUROC = {AUROC_result_noKmer}')

    # plt.figure(1)
    # plt.plot(recall_noKmer, precision_noKmer, label='with kmer', color='r')
    # plt.plot(recall_withKmer, precision_withKmer, label = 'without kmer', color='g')
    # plt.xlim((0, 1))
    # plt.ylim((0.5, 1))
    # plt.xlabel('Sensitivity')
    # plt.ylabel('Precision')
    # plt.legend()
    # print('PR curve')
    # plt.show()

    # plt.figure(2)
    # plt.plot(fpr_withKmer, tpr_withKmer, label='with kmer', color='r')
    # plt.plot(fpr_noKmer, tpr_noKmer, label = 'without kmer', color='g')
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    # plt.xlabel('Specificity')
    # plt.ylabel('Sensitivity')
    # plt.legend()
    # print('ROC curve')
    # plt.show()
