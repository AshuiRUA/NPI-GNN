import networkx as nx
import matplotlib.pyplot as plt
import pickle

# 参数
dataset_name = 'RPI369'
only_Positive = True
draw_plt = False
#######################################################################

if only_Positive == True:
    G_name = f'{dataset_name}_onlyPositive'
else:
    G_name = dataset_name

G_path = f'data/graph/{G_name}/bipartite_graph.edgelist'
G = nx.read_edgelist(G_path)

# 删除节点
# G_temp = G.copy()
# print(len(G_temp.edges))
# e = (str(2090),str(48))
# G_temp.remove_edge(*e)
# print(len(G_temp.edges))

print(type(G))
print(f'number of connected components：{len(list(nx.connected_components(G)))}')

if draw_plt == True:
    plt.subplot(2, 2, 1)
    nx.draw(G, node_size=1)
    plt.show()

# 找到最大的连通分量
# largest_components = max(nx.connected_components(G), key=len)
# print(largest_components)

print('exit\n')