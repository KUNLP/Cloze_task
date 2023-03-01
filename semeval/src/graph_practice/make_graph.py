import os
import pandas as pd
from pandas import DataFrame
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
#의미 트리플 주어-술어-객체
# subj, pred, obj = [], [], []
# with open('../data/transomcs/TransOMCS_full.txt', 'r') as file:
#     for line in file.readlines():
#         #print(line)
#         temp_subj, temp_pred, temp_obj, _ = line.split('\t')
#         subj.append(temp_subj)
#         pred.append(temp_pred)
#         obj.append(temp_obj)
#

# 테스트용 데이터프레임 저장
# relation 20개 종류

rawdata = pd.read_csv('../../data/transomcs/TransOMCS_full.txt', names=['subj', 'pred', 'obj', 'score'], sep='\t')
relation = rawdata['pred'].unique()
input_string = 'university'
condition = (rawdata.subj == input_string) | (rawdata.obj == input_string)
rawdata[condition].to_csv('../data/transomcs/teach_df.csv', sep='\t', index=False)
print(len(relation), relation)
testdf = pd.read_csv('../data/transomcs/teach_df.csv', names=['subj', 'pred', 'obj', 'score'], sep='\t', header=0)
use_df = testdf[:50]

#print(use_df)

G = nx.from_pandas_edgelist(use_df, "subj", "obj", edge_key='pred',edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(18,18))
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos, font_size=24)

plt.show()
print('G의 엣지 개수 {}, 엣지 : {}'.format(G.number_of_nodes(), G.edges))
print('G의 노드 개수 : {}, 노드 : {}'.format(G.number_of_edges(), G.nodes))
#인접행렬
adj_matrix = np.identity(n=G.number_of_nodes())
A = nx.adjacency_matrix(G)
adj_matrix = A.todense()+adj_matrix
#print(adj_matrix)
#print(adj_matrix.shape)
relation2idx = {}
for idx, rel in enumerate(relation):
    relation2idx[rel] = idx
print(relation2idx)


# 특질 행렬 feature matrix row는 노드 개수, column은 선택한 피쳐의 개수 = 여기선 관계가 20개니까 20개로 해보자
feature_matrix = np.zeros((G.number_of_nodes(), len(relation)))
edges = list(G.edges)
# for i in edges:
#     sub, obj, pred = i
#
# for i in range(G.number_of_nodes()):
#     for j in range(len(relation)):

#print(adj_matrix)
#print(adj_matrix2)
#print('요약 : {}'.format(nx.info(G)))
# print('G의 인접행렬 : {}'.format(G.adj))
#
# for key, value in G.adj.items():
#     print(key, value)


def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph.keys():
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath:
                return newpath
    return None

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph.keys():
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph.keys():
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


# graph test
# graph = {'A': ['B', 'C'],
#          'B': ['C', 'D'],
#          'C': ['D'],
#          'D': ['C'],
#          'E': ['F'],
#          'F': ['C']
#      }
#
# print(find_all_paths(graph, 'A', 'D'))