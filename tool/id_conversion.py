
def load_map(path:str):
    map_A_to_B = {}
    with open(path,'r') as fp:
        for line in fp.readlines():
            line = line.strip().split('\t')
            if line[0] != 'NA' and line[1] != 'NA':
                map_A_to_B[line[0]] = line[1]
    return map_A_to_B


def return_A_to_C(map_A_to_B:dict, map_B_to_C:dict):
    map_A_to_C = {}
    for k,v in map_A_to_B.items():
        if v in map_B_to_C:
            map_A_to_C[k] = map_B_to_C[v]
    return map_A_to_C


def write_list(path:str,output_list:list):
    with open(path, 'w') as file:
        for value in output_list:
            file.write(str(value)+'\n')


def write_dict(path:str, output_dict:dict):
    with open(path, 'w') as file:
        for k,v in output_dict.items():
            file.write(f'{k}\t{v}\n')


if __name__ == "__main__":
    A_to_B = load_map(r'tool\data\id_conversion\lncRNA\NPInter2lncRNA_to_modern.txt')
    B_to_C = load_map(r'tool\data\id_conversion\lncRNA\modern_NPInter2.txt')
    A_to_C = return_A_to_C(A_to_B, B_to_C)
    # 输出
    write_dict(r'tool\data\cross_id_NPInter_NPInterLncRNA\lncRNAName_NPInter2LncRNA_NPInter2.txt', A_to_C)