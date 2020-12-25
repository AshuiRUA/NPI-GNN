import sys
import argparse
from torch_geometric.data import DataLoader
import os.path as osp
import os
import torch
import matplotlib.pyplot as plt
import math
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



if __name__ == "__main__":
    # 有kmer的
    # load datset
    print('load dataset with kmer')
    datset_test = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(r'data\dataset\1223_1_inMemory_test_0')
    test_loader = DataLoader(datset_test, batch_size=200)
    # load model
    print('load model with kmer')
    device = torch.device('cuda')
    model = Net_1(datset_test.num_node_features).to(device)
    model.load_state_dict(torch.load(r'result\1223_1\model_0_fold\15'))
    # run test
    AUROC_result_withKmer, fpr_withKmer, tpr_withKmer, precision_withKmer, recall_withKmer = AUROC(model, test_loader, device)

    #没有kmer的
    # load dataset
    print('load dataset without kmer')
    dataset_test = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(r'data\dataset\1223_1_inMemory_noKmer_test_0')
    test_loader = DataLoader(dataset_test, batch_size=200)
    # load model
    print('load model without kmer')
    device = torch.device('cuda')
    model = Net_1(dataset_test.num_node_features).to(device)
    model.load_state_dict(torch.load(r'result\1223_1_noKmer\model_0_fold\35'))
    # run test
    AUROC_result_noKmer, fpr_noKmer, tpr_noKmer, precision_noKmer, recall_noKmer = AUROC(model, test_loader, device)

    print('有kmer')
    print(f'AUROC = {AUROC_result_withKmer}')
    print('没有kmer')
    print(f'AUROC = {AUROC_result_noKmer}')

    plt.figure(1)
    plt.plot(recall_noKmer, precision_noKmer, label='with kmer', color='r')
    plt.plot(recall_withKmer, precision_withKmer, label = 'without kmer', color='g')
    plt.xlim((0, 1))
    plt.ylim((0.5, 1))
    plt.xlabel('Sensitivity')
    plt.ylabel('Precision')
    plt.legend()
    print('PR curve')
    plt.show()

    plt.figure(2)
    plt.plot(fpr_withKmer, tpr_withKmer, label='with kmer', color='r')
    plt.plot(fpr_noKmer, tpr_noKmer, label = 'without kmer', color='g')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.legend()
    print('ROC curve')
    plt.show()
