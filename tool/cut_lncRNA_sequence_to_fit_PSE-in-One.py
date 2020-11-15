import os
import os.path as osp
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="cut lncRNA sequence to fit PSE-in-One")
    parser.add_argument('--input', help='input fasta file')
    parser.add_argument('--output', help='output folder')

    return parser.parse_args()


print('\nstart\n')

args = parse_args()

dataset_name = 'RPI369'

lncRNA_sequence_file_path = args.input
lncRNA_sequence_fasta = open(file=lncRNA_sequence_file_path, mode='r')
lines = lncRNA_sequence_fasta.readlines()

# 创建要输出的文件夹
output_path = args.output
if not osp.exists(path=output_path):
    os.makedirs(output_path)

# 从lncRNA_sequence.fasta中，记录lncRNA名字的行号都找到
indicant_index_list = []    # indicant就是'>'，存储，每个lncRNA序列记录开始的行号
for index in range(len(lines)):
    line = lines[index]
    if line[0] == '>':
        indicant_index_list.append(index)
num_of_lncRNA = len(indicant_index_list)
print('number of lncRNAs', num_of_lncRNA)

#每两个数之间，就是一个lncRNA序列记录
indicant_index_list.append(len(lines))  

# 开始分割
start = 0
end = start + 250
write_count = 1

while end < num_of_lncRNA:
    write_start = indicant_index_list[start]
    write_end = indicant_index_list[end]
    
    name_file = 'lncRNA_sequence' + str(write_count) + '.fasta'
    with open(file=osp.join(output_path, name_file), mode='w') as f:
        i = write_start
        while i < write_end:
            f.write(lines[i])
            i = i + 1
        write_count = write_count + 1
    start = end
    end = end + 250

# 为了防止，因为lncRNA序列记录的个数不能被250整除，而最后剩下不足250个lncRNA序列记录
# 记录剩余的lncRNA序列记录
if num_of_lncRNA % 250 != 0:
    write_start = indicant_index_list[start]
    write_end = len(lines)
    name_file = 'lncRNA_sequence' + str(write_count) + '.fasta'
    with open(file=osp.join(output_path, name_file), mode='w') as f:
        i = write_start
        while i < write_end:
            f.write(lines[i])
            i = i + 1

print('\nexit\n')