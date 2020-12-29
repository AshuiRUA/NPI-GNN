import random
import sys, os

from torch import rand

sys.path.append(os.path.realpath('.'))

from src.generate_edgelist  import read_interaction_dataset, write_interaction_database_reduced

if __name__ == "__main__":
    interaction_dataset_path = 'data/source_database_data/NPInter2.xlsx'
    interaction_list, negative_interaction_list,lncRNA_list, protein_list, lncRNA_name_index_dict, protein_name_index_dict, set_interactionKey, \
        set_negativeInteractionKey = read_interaction_dataset(dataset_path=interaction_dataset_path, dataset_name='NPInter2')
    
    list_interaction_NPInter2_20persent_pureRandom = random.sample(interaction_list, int(len(interaction_list)*0.2))

    write_interaction_database_reduced(r'data\source_database_data\NPInter2_pureRandom_0.20', list_interaction_NPInter2_20persent_pureRandom, [])
