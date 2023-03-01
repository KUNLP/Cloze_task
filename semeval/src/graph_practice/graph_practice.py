import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch


print("#### Graph Init ####")
G = nx.Graph()
DiGraph = nx.DiGraph()

print("#### Add Node to Graph ####")
print("# Add node 1")
G.add_node(1)
print("Num of nodes in G : " + str(G.number_of_nodes()))
print("Graph : "+ str(G.nodes)+'\n')


# 그래프의 전역 군집 계수 (전역 군집 계수 : 전체 그래프에서 군집의 형성 정도, 전역 군집 계수는 각 정점에서의 지역적 군집 계수의 평균)
def getGraphAverageClusteringCoefficient(Graph):
    ccs = []
    for v in Graph.nodes:
        num_connceted_pairs = 0
        for neighbor1 in Graph.neighbors(v):
            for neighbor2 in Graph.neighbors(v):
                if neighbor1 <= neighbor2:
                    continue
                if Graph.has_edge(neighbor1, neighbor2):
                    num_connceted_pairs = num_connceted_pairs + 1
        cc = num_connceted_pairs / (Graph.degree(v) * (Graph.degree(v)-1)/2)
        ccs.append(cc)
    return sum(ccs) / len(ccs)


# 주어진 그래프의 지름을 계산
def getGraphDiameter(Graph):
    diameter = 0
    for v in Graph.nodes:
        length = nx.single_source_shortest_path_length(Graph, v)
        max_length = max(length.values())
        if max_length > diameter:
            diameter = max_length
    return diameter


###########################
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset
from sklearn.metrics import f1_score
import dgl.function as fn


### data 불러오기 Cora 인용 그래프
'''
cora 데이터셋은 2708개의 정점(논문)과 10556개의 간선(인용 관계)로 구성 
각 정점은 1433개의 속성을 가지며 이는 1433개의 단어의 등장 여부를 의미
각 정점은 7개의 유형 중 하나를 가지며 대응되는 논문의 주제를 의미
학습은 140개의 정점을 사용해 이루어진다.
'''
G = CoraGraphDataset()
numClasses = G.num_classes   #7개
G = G[0]
features = G.ndata['feat']    #속성 메트릭스. 정점별 속성
inputFeatureDim = features.shape[1]    # 1433
labels = G.ndata['label']
trainMask = G.ndata['train_mask']
testMask = G.ndata['test_mask']

# SAGEConv 를 정의한거다.
class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.W = nn.Linear(in_feats+in_feats, out_feats, bias=True)  #두 벡터 컨캣하니까

    def forward(self, graph, feature):
        graph.ndata['h'] = feature   #이전 레이어의 임베딩을 가져와서 h라는 이름으로 저장
        graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))  #h를 복사해서 m이라는 이름으로 저장. m을 합계를 내서 neigh로 저장한다.
        degs = graph.in_degrees().to(feature)
        hkNeigh = graph.ndata['neigh'] / degs.unsqueeze(-1)  #합계를 디그리로 나눠줘서 평균을 낸다. 평균은 이웃들로부터 얻은 평균
        hk = self.W(torch.cat((graph.ndata['h'], hkNeigh),  dim=-1))  # 이웃들로얻은 평균과 자기 자신의 이전 레이어 임베딩을 연결해서 신경망을 통과시킨다.
        if self.activation != None:
            hk = self.activation(hk)
        return hk   # 한층의 출력을 얻게 된다.


#GraphSAGE 구현
class GraphSAGE(nn.Module):
    def __init__(self, graph, inFeatDim, numHiddenDim, numClasses, numLayers, activationFunction):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.graph = graph
        self.layers.append(SAGEConv(inFeatDim, numHiddenDim, activationFunction))  # 0번째층
        for i in range(numLayers):
            self.layers.append(SAGEConv(numHiddenDim, numHiddenDim, activationFunction))   # 중간층
        self.layers.append(SAGEConv(numHiddenDim, numClasses, activation=None)) # 마지막층. 마지막층에선 액티베이션 사용 않함

    def forward(self, features):
        x = features
        for layer in self.layers:
            x = layer(self.graph, x)
        return x

model = GraphSAGE(G, inputFeatureDim, numHiddenDim, numClasses, numLayers, F.relu)


def train(model, lossFunction, features, labels, trainMask, optimizer, numEpochs):
    for epoch in range(numEpochs):
        model.train()
        logits = model(features)
        loss = lossFunction(logits[trainMask], labels[trainMask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

lossFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
train(model, lossFunction, features, labels, trainMask, optimizer, numEpochs)


def evaluateTest(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        macro_f1 = f1_score(labels, indices, average='macro')
        correct = torch.sum(indices==labels)
        return correct.item()*1.0/len(labels), macro_f1

def test(model, features, labels, testMask):
    acc, macro_f1 = evaluateTest(model, features, labels, testMask)

test(model, features, labels, testMask)