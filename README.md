# GNN predict rpi
In this work, by integrating SEAL, GraphSAGE and top-k pooling, we construct a GNN based model to predict ncRNA-protein interactions.

### Dependency:
Python 3.6

pytorch 1.4.0

torch-geometric 1.4.2

### Usage:
First, you have to generate dataset for training.
>Python generate_dataset.py --datasetName NPInter2 --hopNumber 2 --node2vecWindowSize 5

This project did not provided node2vec result for all window size, you may have to use node2vec-master/src/main to generate result following hint which is outputted by generate_dataset.py.

Then, train GNN and save models.
>Python train.py --name object_name --datasetName NPInter2 --hopNumber 2 --node2vecWindowSize 5

