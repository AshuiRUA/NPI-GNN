import sys
import argparse
from torch_geometric.data import DataLoader
import os.path as osp
import os
import torch
sys.path.append(r"C:\Python_prj\GNN_predict_rpi_0930")

from src.classes import LncRNA_Protein_Interaction_dataset, Net_1
from src.methods import Accuracy_Precision_Sensitivity_Specificity_MCC



def parse_args():
    parser = argparse.ArgumentParser(description="load model and test dataset to run test")
    parser.add_argument('--testName', help='name of this test')
    parser.add_argument('--path_model', help='the path of model')
    parser.add_argument('--path_dataset', help='the path of test dataset')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # load datset
    print('load dataset')
    datset_test = LncRNA_Protein_Interaction_dataset(args.path_dataset)
    test_loader = DataLoader(datset_test, batch_size=60)
    # load model
    print('load model')
    device = torch.device('cuda')
    model = Net_1(datset_test.num_node_features).to(device)
    model.load_state_dict(torch.load(args.path_model))
    # run test
    print('run test')
    Accuracy, Precision, Sensitivity, Specificity, MCC = Accuracy_Precision_Sensitivity_Specificity_MCC(model, test_loader, device)
    output = 'Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(Accuracy, Precision, Sensitivity, Specificity, MCC)
    print(output)
    # output log
    path_log_folder = osp.join(r'.\result', args.testName)
    if not osp.exists(path_log_folder):
        os.mkdir(path_log_folder)
    path_log = osp.join(path_log_folder, 'log.txt')
    with open(path_log, 'w') as log:
        log.write(output)
