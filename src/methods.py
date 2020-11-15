import pickle

import sys, os
sys.path.append(os.path.realpath('.'))
sys.path.append(r"C:\Python_prj\GNN_predict_rpi_0930")


import sys, os
sys.path.append(os.path.realpath('.'))

import sys, os
sys.path.append(os.path.realpath('.'))

import sys, os
sys.path.append(os.path.realpath('.'))
from src.classes import  LncRNA_Protein_Interaction


def nodeSerialNumber_listIndex_dict_generation(node_list):
    node_name_index_dict = {}
    for index in range(len(node_list)):
        node_name_index_dict[node_list[index].serial_number] = index
    return node_name_index_dict


def reset_basic_data(interaction_list, negative_interaction_list, lncRNA_list, protein_list):
    for lncRNA in lncRNA_list:
        lncRNA.interaction_list = []
    for protein in protein_list:
        protein.interaction_list = []
    
    lncRNA_serial_number_index_dict = nodeSerialNumber_listIndex_dict_generation(lncRNA_list)
    protein_serial_number_index_dict = nodeSerialNumber_listIndex_dict_generation(protein_list)

    new_interaction_list = []
    for i in range(len(interaction_list)):
        interaction = interaction_list[i]
        lncRNA_index = lncRNA_serial_number_index_dict[interaction.lncRNA.serial_number]
        protein_index = protein_serial_number_index_dict[interaction.protein.serial_number]

        temp_interaction = LncRNA_Protein_Interaction(lncRNA=lncRNA_list[lncRNA_index], protein=protein_list[protein_index], y=1)
        new_interaction_list.append(temp_interaction)
        lncRNA_list[lncRNA_index].interaction_list.append(temp_interaction)
        protein_list[protein_index].interaction_list.append(temp_interaction)
    
    new_negative_interaction_list = []
    for i in range(len(negative_interaction_list)):
        interaction = negative_interaction_list[i]
        lncRNA_index = lncRNA_serial_number_index_dict[interaction.lncRNA.serial_number]
        protein_index = protein_serial_number_index_dict[interaction.protein.serial_number]

        temp_interaction = LncRNA_Protein_Interaction(lncRNA=lncRNA_list[lncRNA_index], protein=protein_list[protein_index], y=0)
        new_negative_interaction_list.append(temp_interaction)
        lncRNA_list[lncRNA_index].interaction_list.append(temp_interaction)
        protein_list[protein_index].interaction_list.append(temp_interaction)
 
    return new_interaction_list, new_negative_interaction_list, lncRNA_list, protein_list


def load_intermediate_products(project_name):
    interaction_list_path = f'data/intermediate_products/{project_name}/interaction_list.txt'
    negative_interaction_list_path = f'data/intermediate_products/{project_name}/negative_interaction_list.txt'
    lncRNA_list_path = f'data/intermediate_products/{project_name}/lncRNA_list.txt'
    protein_list_path = f'data/intermediate_products/{project_name}/protein_list.txt'
    with open(file=interaction_list_path, mode='rb') as f:
        interaction_list = pickle.load(f)
    with open(file=negative_interaction_list_path, mode='rb') as f:
        negative_interaction_list = pickle.load(f)
    with open(file=lncRNA_list_path, mode='rb') as f:
        lncRNA_list = pickle.load(f)
    with open(file=protein_list_path, mode='rb') as f:
        protein_list = pickle.load(f)
    # 重新建立node和interaction的相互包含的关系
    return reset_basic_data(interaction_list, negative_interaction_list, lncRNA_list, protein_list)\


def Accuracy(model, loader, device):
    model.eval()
    
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim = 1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def Accuracy_Precision_Sensitivity_Specificity_MCC(model, loader, device):
    model.eval()
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim = 1)[1]
        for index in range(len(pred)):
            if pred[index] == 1 and data.y[index] == 1:
                TP += 1
            elif pred[index] == 1 and data.y[index] == 0:
                FP += 1
            elif pred[index] == 0 and data.y[index] == 1:
                FN += 1
            else:
                TN += 1
    print('TP: %d, FN: %d, TN: %d, FP: %d' % (TP, FN, TN, FP))
    if (TP + TN + FP + FN) != 0:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        Accuracy = 0
    if (TP + FP) != 0:
        Precision = (TP) / (TP + FP)
    else:
        Precision = 0
    if (TP + FN) != 0:
        Sensitivity = (TP) / (TP + FN)
    else:
        Sensitivity = 0
    if (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) != 0:
        MCC = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
    else:
        MCC = 0
    if (FP + TN) != 0:
        Specificity = TN / (FP + TN)
    else:
        Specificity = 0
    return Accuracy, Precision, Sensitivity, Specificity, MCC

# def MCC(model, loader, device):
#     model.eval()
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for data in loader:
#         data = data.to(device)
#         pred = model(data).max(dim = 1)[1]
#         for index in range(len(pred)):
#             if pred[index] == 1 and data.y[index] == 1:
#                 TP += 1
#             elif pred[index] == 1 and data.y[index] == 0:
#                 FP += 1
#             elif pred[index] == 0 and data.y[index] == 1:
#                 FN += 1
#             else:
#                 TN += 1
#     MCC = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)
#     return MCC


def test_lncRNA_list(lncRNA_list):
    for lncRNA in lncRNA_list:
        for interaction in lncRNA.interaction_list:
            if interaction.lncRNA.serial_number != lncRNA.serial_number:
                print('error')


def test_protein_list(protein_list):
    for protein in protein_list:
        for interaction in protein.interaction_list:
            if interaction.protein.serial_number != protein.serial_number:
                print('error')


def dataset_analysis(dataset):
    dict_label_dataNumber = {}
    for data in dataset:
        label = int(data.y)
        if label not in dict_label_dataNumber:
            dict_label_dataNumber[label] = 1
        else:
            dict_label_dataNumber[label] = dict_label_dataNumber[label] + 1
    print(dict_label_dataNumber)


def average_list(list_input):
    average = 0
    for i in range(len(list_input)):
        average = (average * i + list_input[i]) / (i + 1)
    return average


def nodeName_listIndex_dict_generation(node_list):
    node_name_index_dict = {}
    for index in range(len(node_list)):
        node_name_index_dict[node_list[index].name] = index
    return node_name_index_dict
