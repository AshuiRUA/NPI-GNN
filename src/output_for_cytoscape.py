import os.path as osp
import os,sys
import argparse
import random
from openpyxl import load_workbook

sys.path.append(os.path.realpath('.'))
from src.classes import LncRNA
from src.classes import Protein
from src.classes import LncRNA_Protein_Interaction, LncRNA_Protein_Interaction_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--datasetName', default='RPI369', help='raw interactions dataset')

    return parser.parse_args()


def read_interaction_dataset(dataset_path, dataset_name):
    global interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, protein_name_index_dict
    # lncRNA_name_index_dict, protein_name_index_dic为了在interaction dataset中，读到重复的lncRNA或protein时
    # 能在lncRNA_list和protein_list中快速的找到
    if not osp.exists(dataset_path):
        raise Exception('interaction datset does not exist')
    print('开始读取xlsx文件')
    wb = load_workbook(dataset_path)
    sheets = wb.worksheets   # 获取当前所有的sheet
    sheet = sheets[0]
    rows = sheet.rows

    serial_number = 0
    lncRNA_count = 0
    protein_count = 0
    flag = 0 #用来排除RPI7317.xlsx的第一行

    for row in rows:
        #排除第一行
        if flag == 0:
            flag = flag + 1
            continue
        #读出这一行的每个元素,每一行对应一个interaction实例，如果这个interaction对应的lncRNA和protein还没创建，就创建它
        #并在索引词典中加入它在lncRNA_list或者protein_list中的索引
        [lncRNA_name, protein_name, label] = [col.value for col in row]
        label = int(label)
        if lncRNA_name not in lncRNA_name_index_dict:   # 新的，没创建过的lncRNA
            temp_lncRNA = LncRNA(lncRNA_name, serial_number, 'LncRNA')
            lncRNA_list.append(temp_lncRNA)
            lncRNA_name_index_dict[lncRNA_name] = lncRNA_count
            serial_number = serial_number + 1
            lncRNA_count = lncRNA_count + 1
        else:   # 在interaction dataset中已经读到过，已经创建了对象的lncRNA，就存在lncRNA_list中
            temp_lncRNA = lncRNA_list[lncRNA_name_index_dict[lncRNA_name]]
        if protein_name not in protein_name_index_dict: # 新的，没创建过的protein
            temp_protein = Protein(protein_name, serial_number, 'Protein')
            protein_list.append(temp_protein)
            protein_name_index_dict[protein_name] = protein_count
            serial_number = serial_number + 1
            protein_count = protein_count + 1
        else:   # 在interaction dataset中已经读到过，已经创建了对象的protein，就存在protein_list中
            temp_protein = protein_list[protein_name_index_dict[protein_name]]
        temp_interaction = LncRNA_Protein_Interaction(temp_lncRNA, temp_protein, label)
        # print(temp_interaction.protein.name, temp_interaction.lncRNA.name)
        
        if label == 1:
            interaction_list.append(temp_interaction)
        elif label == 0:
            negative_interaction_list.append(temp_interaction)
        else:
            print(label)
            raise Exception('{dataset_name}中有除了0和1之外的label'.format(dataset_name=dataset_name))

        temp_lncRNA.interaction_list.append(temp_interaction)
        temp_protein.interaction_list.append(temp_interaction)
    print('读入的lncRNA总数：{:d}, 读入的protein总数：{:d}, node总数：{:d}'.format(lncRNA_count, protein_count, lncRNA_count + protein_count))
    print(f'读入的正样本数：{len(interaction_list)}, 读入的负样本数：{len(negative_interaction_list)}\n')


def negative_interaction_generation():
    global lncRNA_list, protein_list, interaction_list, negative_interaction_list

    if len(negative_interaction_list) != 0:
        raise Exception('negative interactions exist')

    num_of_interaction = len(interaction_list)
    num_of_lncRNA = len(lncRNA_list)
    num_of_protein = len(protein_list)

    negative_interaction_count = 0
    while(negative_interaction_count < num_of_interaction):
        random_index_lncRNA = random.randint(0, num_of_lncRNA - 1)
        random_index_protein = random.randint(0, num_of_protein - 1)
        temp_lncRNA = lncRNA_list[random_index_lncRNA]
        temp_protein = protein_list[random_index_protein]
        # 检查随机选出的lncRNA和protein是不是有已知相互作用
        repetitive_interaction = 0
        for interaction in temp_lncRNA.interaction_list:
            if interaction.protein.serial_number == temp_protein.serial_number:
                repetitive_interaction = 1
                break
        if repetitive_interaction == 1:
            continue
        # 经过检查，随机选出的lncRNA和protein是可以作为负样本的
        temp_interaction = LncRNA_Protein_Interaction(temp_lncRNA, temp_protein, 0)
        negative_interaction_list.append(temp_interaction)
        temp_lncRNA.interaction_list.append(temp_interaction)
        temp_protein.interaction_list.append(temp_interaction)
        negative_interaction_count = negative_interaction_count + 1
    print('生成了', len(negative_interaction_list), '个负样本')
    return negative_interaction_list


args = parse_args()

interaction_list = []
negative_interaction_list = []
lncRNA_list = []
protein_list = []
lncRNA_name_index_dict = {}
protein_name_index_dict = {}

interaction_dataset_path = 'data/source_database_data/'+ args.datasetName + '.xlsx'
read_interaction_dataset(dataset_path=interaction_dataset_path, dataset_name=args.datasetName)

if  not (args.datasetName == 'RPI2241' or args.datasetName == 'RPI369'):
    negative_interaction_list = negative_interaction_generation() # 生成负样本

path_output = './data/cytoscape_graph'
if not osp.exists(path_output):
    os.makedirs(path_output)

path_file_output = osp.join(path_output, args.datasetName) + '.txt'
file_output = open(path_file_output, mode='w')
file_output.write('source_node\ttarget_node\tSource Node Attribute\tTarget Node Attribute\n')

for interaction in interaction_list:
    file_output.write(f'{interaction.lncRNA.name}\t{interaction.protein.name}\tlncRNA\tprotein\n')
# for interaction in negative_interaction_list:
#     file_output.write(f'{interaction.lncRNA.name}\t{interaction.protein.name}\tlncRNA\tprotein\n')