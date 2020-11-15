# NPI-GNN
In this work, by integrating SEAL, GraphSAGE and top-k pooling, we construct a GNN based model to predict ncRNA-protein interactions.

### Dependency:
Python 3.6

pytorch 1.4.0

torch-geometric 1.4.2

### Usage:
#### 1.Generating edgelist for node2vec.
>Python .\src\generate_edgelist.py --projectName yourProjectName --interactionDatasetName NPInter2 

This will output a edgelist file in 'data/graph/***yourProjectName***/bipartite_graph.edgelist' and an empty folder 'data/node2vec_result/***yourProjectName***' to store node2vec result.

**Necessary parameters**
* --projectName : The name of dataset you want to generate.
* --interactionName : The name existing ncRNA-protein interaction dataset, like NPInter2, NPInter2_lncRNA, RPI7317,RPI2241, and RPI369.

**Optional parameters**
* --createBalanceDataset : Default = 1. Creating balance dataset for unbalanced interaction dataset.
* --reduce: Default = 0. Whether you want to randomly reducing the interaction database, and also maintain one connected component. *1* means yes, *0* means no.
* --reduceRatio: Default = 0.5. The reduce Ratio of reduced dataset.

#### 2. Running node2vec.
>Python .\node2vec-master\src\main.py --input 'data/graph/yourProjectName/bipartite_graph.edgelist' --output 'data/node2vec_result/yourProjectName/result.emb' --window-size 5

**Necessary parameters**
* --input : The input edgelist file generated by '.\src\generate_edgelist.py'.
* --output : The output node2vec result. This parameter has a pattern: 'data/node2vec_result/***yourProjectName***/result.emb'. 
The name of the output file must be result.emb, and the folder it in is the folder generated by '.\src\generate_edgelist.py'

**Optional parameters**

* please see <https://github.com/aditya-grover/node2vec>

#### 3. Generating dataset for training.
>Python src/generate_dataset.py --projectName yourProjectName --interactionDatasetName NPInter2 --inMemory 0

**Necessary parameters**
* --projectName : The name of dataset you want to generate.
* --interactionName : The name existing ncRNA-protein interaction dataset, like NPInter2, NPInter2_lncRNA, RPI7317,RPI2241, and RPI369.
* --inMemory: Whether you want to generate a in Memory dataset for GNN, *1* means yes, *0* means no.

**Optional parameters**
* --hopNumber：Default = 2. The hop number of *h*-hop local enclosing subgraph.
* --shuffle: Default = 1. Shuffle interactions before generate dataset, *1* means yes, *0* means no.
* --noKmer: Default = 0. If you don't want to add *k*-mer frequencies into node feature, set it to *1*;

#### 4. Running cross validation of NPI-GNN  and save models.
>Python .\src\train.py --trainingName nameOfTraining --datasetName yourProjectName --interactionDatasetName NPInter2 --epochNumber 50 --inMemory 0

This will save modules and training log in 'result/yourProjectName'

**Necessary parameters**
* --trainingName: The name of this training.
* --datasetName: The name of dataset you want to use, and this is the same as ***yourProjectName*** in previous steps.
* --interactionName : The name existing ncRNA-protein interaction dataset, like NPInter2, NPInter2_lncRNA, RPI7317,RPI2241, and RPI369.
* --inMemory: Whether you want to generate a in Memory dataset for GNN, *1* means yes, *0* means no.

**Optional parameters**
* --crossValidation: Default = 1. Whether you want to do a cross Validation, *1* means yes, *0* means no.
* --foldNumber: Default = 5, The number of folds of the cross validation.
* --epochNumber: Default = 50. The number of epoch of each fol
* --hopNumber：Default = 2. The hop number of *h*-hop local enclosing subgraph.
* --initialLearningRate: Default = 0.005. The initial learning rate of this training.
* --l2WeightDecay: Default = 0.0005. The L2 weight decay of this training.
* --batchSize: Default = 60. The batch size of this training.
