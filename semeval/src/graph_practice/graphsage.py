import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset
from dgl.nn.pytorch.conv import SAGEConv
from sklearn.metrics import f1_score
from tqdm import tqdm


G = CoraGraphDataset()
numClasses = G.num_classes
G = G[0]
features = G.ndata['feat']
inputFeatureDim = features.shape[1]
labels = G.ndata['label']
trainMask = G.ndata['train_mask']
testMask = G.ndata['test_mask']


class GraphSAGE(nn.Module):
    def __init__(self, graph, inFeatDim, numHiddenDim, numClasses, numLayers, activationFunction, dropoutProb, aggregatorType):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.graph = graph
        self.layers.append(SAGEConv(inFeatDim, numHiddenDim, aggregatorType, dropoutProb, activationFunction))

        for i in range(numLayers):
            self.layers.append(SAGEConv(numHiddenDim, numHiddenDim, aggregatorType, dropoutProb, activationFunction))
        self.layers.append(SAGEConv(numHiddenDim, numClasses, aggregatorType, dropoutProb, activation=None))

    def forward(self, features):
        x = features
        for layer in self.layers:
            x = layer(self.graph, x)
        return x


def evaluateTest(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        macro_f1 = f1_score(labels, indices, average='macro')
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 /len(labels), macro_f1


def test(model, features, labels, testMask):
    acc, macro_f1 = evaluateTest(model, features, labels, testMask)
    return acc, macro_f1


def train(model, lossFunction, features, labels, trainMask, optimizer, numEpochs):
    for epoch in range(numEpochs):
        model.train()
        logits = model(features)
        loss = lossFunction(logits[trainMask], labels[trainMask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(test(model,features, labels,testMask))




numHiddenDim = 768
numLayers = 3
dropoutProb = 0.1
aggregatorType = 'mean'
learningRate = 2e-5
weightDecay = 0
numEpochs = 10

model = GraphSAGE(G, inputFeatureDim, numHiddenDim, numClasses, numLayers, F.relu, dropoutProb, aggregatorType)

lossFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
train(model, lossFunction, features, labels, trainMask, optimizer, numEpochs)
print(labels)