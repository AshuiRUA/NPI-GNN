# NPI-GNN
In this work, by integrating SEAL, GraphSAGE and top-k pooling, we construct a GNN based model to predict ncRNA-protein interactions.

### Dependency:
Python 3.6

pytorch 1.4.0

torch-geometric 1.4.2

### Usage:
First, you have to generate edgelist for node2vec.
>Python .\src\generate_edgelist.py --projectName yourProjectName --interactionDatasetName NPInter2 --createBalanceDataset 1 
<<<<<<< HEAD
<<<<<<< HEAD

push test
=======
>>>>>>> 08847d0... 20201018 有了in memory数据集，测试后提交
=======
>>>>>>> 4216767b1450b7b85fe4974fc54f997ec4fee435

push test

This will output a edgelist file in 'data/graph/yourProjectName/bipartite_graph.edgelist' and an empty folder 'data/node2vec_result/yourProjectName' to store node2vec result.

Second, you have to run node2vec.
>Python .\node2vec-master\src\main.py --input 'data/graph/yourProjectName/bipartite_graph.edgelist' --output 'data/node2vec_result/yourProjectName/result.emb' --window-size 5

You can use different window size, but please remember it.

Third, generate dataset for training.
>Python src/generate_dataset.py --projectName yourProjectName --interactionDatasetName NPInter2 --inMemory 0

Finally, run cross validation of NPIGNN  and save models.
>Python .\src\train.py --trainingName nameOfTraining --datasetName yourProjectName --interactionDatasetName NPInter2 --node2vecWindowSize 5  --epochNumber 50 --inMemory 0

This will save modules and training log in 'result/yourProjectName'

If you want to reduce the size of the dataset while keeping the bipartite graph has only one connected component, you need to use "--reduce" and "--reduceRatio" on generate_list.py
>Python .\src\generate_edgelist.py --projectName yourProjectName --interactionDatasetName NPInter2 --createBalanceDataset 1 --reduce 1 --reduceRatio 0.5

If you don't want to use k-mer, you need ro use "noK-mer" on generate_dataset.py
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>Python src/generate_dataset.py --projectName yourProjectName --interactionDatasetName NPInter2 --inMemory 0 --noKmer 1
=======
>Python src/generate_dataset.py --projectName yourProjectName --interactionDatasetName NPInter2 --inMemory 0 --noKmer 1
>>>>>>> 08847d0... 20201018 有了in memory数据集，测试后提交
=======
>Python src/generate_dataset.py --projectName yourProjectName --interactionDatasetName NPInter2 --inMemory 0 --noKmer 1
>>>>>>> bdc6081... Update README.md
=======
>Python src/generate_dataset.py --projectName yourProjectName --interactionDatasetName NPInter2 --inMemory 0 --noKmer 1
>>>>>>> 73fcb72... Update README.md
=======
>Python src/generate_dataset.py --projectName yourProjectName --interactionDatasetName NPInter2 --inMemory 0 --noKmer 1
>>>>>>> 4216767b1450b7b85fe4974fc54f997ec4fee435
