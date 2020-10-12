# GNN predict rpi
In this work, by integrating SEAL, GraphSAGE and top-k pooling, we construct a GNN based model to predict ncRNA-protein interactions.

### Dependency:
Python 3.6

pytorch 1.4.0

torch-geometric 1.4.2

### Usage:
First, you have to generate edgelist for node2vec.
>Python .\src\generate_edgelist.py --projectName yourProjectName --datasetName NPInter2 --createBalanceDataset True 

This will output a edgelist file in 'data/graph/yourProjectName/bipartite_graph.edgelist' and an empty folder 'data/node2vec_result/yourProjectName' to store node2vec result.

Second, you have to run node2vec.
>Python .\node2vec-master\src\main.py --input 'data/graph/yourProjectName/bipartite_graph.edgelist' --output 'data/node2vec_result/yourProjectName/result.emb' --window-size 5

You can use different window size, but please remember it.

Third, generate dataset for training.
>Python src/generate_dataset.py --projectName yourProjectName --datasetName NPInter2 --hopNumber 2 --shuffle True

Finally, train GNN and save models.
>Python .\src\train.py --projectName yourProjectName --datasetName NPInter2 --hopNumber 2 --node2vecWindowSize 5 --crossValidation True --foldNumber 5 --epochNumber 50 --initialLearningRate 0.005 --l2WeightDecay 0.0005

This will save modules and training log in 'result/yourProjectName'

If you want to reduce the size of the dataset while keeping the bipartite graph has only one connected component, you need to use "--reduce" and "--reduceRatio" on generate_list.py
>Python .\src\generate_edgelist.py --projectName yourProjectName --datasetName NPInter2 --createBalanceDataset True --reduce True --reduceRatio 0.5

If you don't want to use k-mer, you need ro use "noK-mer" on generate_dataset.py
>Python src/generate_dataset.py --projectName yourProjectName --datasetName NPInter2 --hopNumber 2 --shuffle True --noKmer True