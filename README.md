# NPI-GNN

In this work, by integrating SEAL, GraphSAGE and top-k pooling, we construct a GNN based model to predict ncRNA-protein interactions.

### Dependency

Python 3.6

pytorch 1.4.0

torch-geometric 1.4.2

dependencies for node2vec
    
    Note: This project contains node2vec released in https://github.com/aditya-grover/node2vec. So you need to configure its dependencies according to its description. But this node2vec is based on Python2, and some of its dependencies are also based on Python2. The code for node2vec included in the project was modified to make it work under Python3.
    
    Note: We strongly recommend that you configure a Python 2 environment to run node2vec and download the source code from this site.This will avoid a lot of confusion with Python2 and Python3.

### Workflow

#### 1.Generating edgelist for node2vec

>Python .\src\generate_edgelist.py --projectName yourProjectName --interactionDatasetName NPInter2

This will output five edgelist file in 'data/graph/{yourProjectName}/training_{0-4}/bipartite_graph.edgelist' and five empty folder 'data/node2vec_result/{yourProjectName}/training_{0-4}' to store node2vec result.

##### Necessary parameters**

* --projectName : The name of dataset you want to generate.
* --interactionName : The name of existing ncRNA-protein interaction dataset, like NPInter2, NPInter2_lncRNA, RPI7317,RPI2241, and RPI369.

##### Optional parameters**

* --createBalanceDataset : Default = 1. Creating balance dataset for unbalanced interaction dataset.
* --reduce: Default = 0. Whether you want to randomly reducing the interaction database, and also maintain one connected component. *1* means yes, *0* means no.
* --reduceRatio: Default = 0.5. The reduce Ratio of reduced dataset.

#### 2. Running node2vec for each fold

>Python .\node2vec-master\src\main.py --input 'data/graph/yourProjectName/training_{0-4}/bipartite_graph.edgelist' --output 'data/node2vec_result/yourProjectName/training{0-4}/result.emb'

##### Necessary parameters

* --input : The input edgelist file generated by '.\src\generate_edgelist.py'.
* --output : The output node2vec result. This parameter has a pattern: 'data/node2vec_result/{yourProjectName}/result.emb'.
The name of the output file must be result.emb, and the folder it in is the folder generated by '.\src\generate_edgelist.py'

##### Optional parameters

* please see <https://github.com/aditya-grover/node2vec>

#### 3. Generating dataset for training

>Python .\src\generate_dataset.py --projectName yourProjectName --interactionDatasetName NPInter2 --fold 0

Note: There is randomness in the existence process because the data is shuffled as the dataset is generated.

##### Necessary parameters

* --projectName : The name of dataset you want to generate.
* --interactionName : The name of existing ncRNA-protein interaction dataset, like NPInter2, NPInter2_lncRNA, RPI7317,RPI2241, and RPI369.
* --fold : Generating dataset for which fold, 0-4


##### Optional parameters

* --createBalanceDataset : Default = 1. This parameter must keep it as same as you run src/generate_edgelist.py
* --shuffle: Default = 1. Shuffle interactions before generate dataset, *1* means yes, *0* means no.
* --noKmer: Default = 0. If you don't want to add *k*-mer frequencies into node feature, set it to *1*;

#### 4. Running cross validation of NPI-GNN  and save models.

>Python .\src\train_with_twoDataset.py --trainingName nameOfTraining --datasetName yourProjectName --interactionDatasetName NPInter2 --epochNumber 50 --fold 0

This will save modules and training log in 'result/{yourProjectName}'.Every five epochs report the model performance and save the model. Please note that the paper uses the performance of the model with the largest mean MCC in a cross-validation. 

Note: There is randomness in the existence process because the data is shuffled as the dataset is read.

Note: The initial learning rate was 0.0001 when we perform 5-fold cross validation on RPI369. Moreover, in the data set of RPI369, the model often fails to converge, which further proves that NPI-GNN is not suitable to run on the data set with too small local subgraphs.

##### Necessary parameters

* --trainingName: The name of this training.
* --datasetName: The name of dataset you want to use, and this is the same as {yourProjectName} in previous steps.
* --interactionName : The name of existing ncRNA-protein interaction dataset, like NPInter2, NPInter2_lncRNA, RPI7317,RPI2241, and RPI369.
* --fold : Which fold you want to run, 0-4.

##### Optional parameters

* --epochNumber: Default = 50. The number of epoch of each fold.
* --hopNumber：Default = 2. The hop number of *h*-hop local enclosing subgraph.
* --initialLearningRate: Default = 0.001. The initial learning rate of this training.
* --l2WeightDecay: Default = 0.001. The L2 weight decay of this training.
* --batchSize: Default = 200. The batch size of this training.

### How to use your own interaction dataset

#### 1.Prepare data

* Make you interaction dataset an xlsx file : {yourInteractionDataset}.xlsx. The format of this xlsx file, please refer to existing xlsx in 'data/source_database_data'.

* Prepare lncRNA 3-mer frequencies result : lncRNA_3_mer.txt, and put it under 'data/lncRNA_3_mer/{yourInteractionDataset}'.

* Prepare protein fasta file, create new folder '{yourInteractionDataset}' under 'data/protein_2_mer/', use 'tool/protein_2-mer_generation.py' to generate protein 2-mer result

>Python ./tool/protein_2-mer_generation.py --input_fasta_file 'Path_of_you_protein_fasta_file' --output_folder 'data/protein_2_mer/yourInteractionDataset' --k 2

##### Necessary parameters

* --input_fasta_file: The path of your protein fasta file.
* --output_folder : The folder you created : 'data/protein_2_mer/{yourInteractionDataset}'

##### Optional parameters

* --k: Default = 2. The *k* of *k*-mer frequency.

#### 2. Run  workflow

When running the workflow using you on interaction dataset, all **--interactionDataset** have to be {yourInteractionDataset}.

### How to predict novel interaction based on existing data

#### 1.  Complete a Workflow 1-4 or select a completed Workflow 1-4

#### 2. run src/case_study_negativeSample.py

>python ./src/case_study_negativeSample.py --caseStudyName nameOfThisCaseStudy --projectName yourProjectName --fold 0 --interactionDatasetName NPInter2 --createBalanceDataset 1 --modelPath PathOfTheModel --threshold 0.95

This will create a folder in 'data/case_study/{nameOfThisCaseStudy}. 

##### Necessary parameters

* --caseStudyName : The name of this case study.
* --projectName : The project Name in workflow 1-4
* --fold : 0-4, select one fold of the 5-fold cross-validation and complete the case study based on it.
* --interactionDatasetName : The name of existing ncRNA-protein interaction dataset, like NPInter2, NPInter2_lncRNA, RPI7317,RPI2241, and RPI369. Please keep it the same as you run workflow 1-4.
* --createBalanceDataset : 0 or 1, be consistent with workflow 1-4
* --modelPath : The path of trained NPI-GNN model
* --threshold : The threshold that determines the sample to be positive

##### Optional parameters

* --noKmer : 0 or1, be consistent with workflow 1-4.

### How to run leave-one-out cross validation using interactions shared by NPInter2 and RPI2241

#### 1. Generate edgelist for node2vec based on NPInter2

> python ./src/generate_edgelist_NPInter2_RPI2241_mutual_interaction_study.py --projectName yourProjectName_1 --path_set_negativeInteractionKey yourNegativeInteractionKeyPath

This will generate a edgelist under 'data/graph/{yourProjectName_1}/bipartite_graph.edgelist' and a folder 'data/node2vec_result/{yourProjectName_1}'

##### Necessary parameters

* --projectName : the name of this project

##### Optional parameters

* --path_set_negativeInteractionKey: default = 'data/set_allInteractionKey/1223_1/set_negativeInteractionKey_all', if you want to use other randomly selected negative sample, just change '1223_1' to other project name. But the project you choose must be based on NPInter2.

#### 2. run node2vec

>python ./node2vec-master/src/main.py --input data/graph/{yourProjectName_1}/bipartite_graph.edgelist -- output data/node2vec_result/{yourProjectName_1}/result.emb


#### 3. run leave-one-out cross validation based on NPInter2

> ./run_mutualInteraction_NPInter2.bat yourProjectName_1 1223_1

When generating edgelist, if you don't use the default negative sample, 1223_1 should be changed to the corresponding project name

#### 4. Generate edgelist for node2vec based on RPI2241

> python ./src/generate_edgelist_NPInter2_RPI2241_mutual_interaction_study.py --projectName yourProjectName_2 -- interactionDatasetName RPI2241 --createBalanceDataset 0 

This will generate a edgelist under 'data/graph/{yourProjectName_2}/bipartite_graph.edgelist' and a folder 'data/node2vec_result/{yourProjectName_2}'

##### Necessary parameters

* --projectName : the name of this project, the project name here should be different from yourProjectName_1 in Step 1 and 2
* --interactionDatasetName : must be RPI2241
* --createBalanceDataset must be 0

#### 5. run node2vec

>python ./node2vec-master/src/main.py --input data/graph/{yourProjectName_2}/bipartite_graph.edgelist -- output data/node2vec_result/{yourProjectName_2}/result.emb

#### 6. run leave-one-out cross validation based on RPI2241

>.\run_mutualInteraction_RPI2241.bat yourProjectName_2


### How to reduce NPInter dataset

>python ./src/generate_edgelist.py --projectName nameOfThisReduction --reduce 1 --reduceRatio 0.25

This will generate a reduced NPInter2 at 'data/source_database_data/NPInter2_{reduceRatio}.xlsx'
##### Necessary parameters

* --projectName : The name of this reduction
* --reduce : must be 1
* --reduceRatio : The reduce ratio you want to use

### How to use NPI-GNN predict know interaction

#### 1.  Complete a Workflow 1-4 or select a completed Workflow 1-4

#### 2. run src/case_study.py

>python ./src/case_study.py --caseStudyName nameOfThisCaseStudy --projectName yourProjectName --fold 0 --interactionDatasetName NPInter2 --createBalanceDataset 1 --modelPath PathOfTheModel

This will create a folder in 'data/case_study/{nameOfThisCaseStudy}

##### Necessary parameters

* --caseStudyName : The name of this case study.
* --projectName : The project Name in workflow 1-4
* --fold : 0-4, select one fold of the 5-fold cross-validation and complete the case study based on it.
* --interactionDatasetName : The name of existing ncRNA-protein interaction dataset, like NPInter2, NPInter2_lncRNA, RPI7317,RPI2241, and RPI369. Please keep it the same as you run workflow 1-4.
* --createBalanceDataset : 0 or 1, be consistent with workflow 1-4
* --modelPath : The path of trained NPI-GNN model

##### Optional parameters

* --noKmer : 0 or1, be consistent with workflow 1-4

# Citation

[1] Zi-Ang Shen, Tao Luo, Yuan-Ke Zhou, Han Yu, Pu-Feng Du, NPI-GNN: Predicting ncRNA–protein interactions with deep graph neural networks, Briefings in Bioinformatics, Volume 22, Issue 5, September 2021, bbab051, https://doi.org/10.1093/bib/bbab051