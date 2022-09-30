from _csv import reader
import random

import numpy as np

import torch

import GN


def load_data():
    data = []
    global max_node
    max_node = 7
    with open('data/CHEM_Data.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        j = 0
        for row in csv_reader:
            if j != 0:
                data.append(row)
            j += 1

    random.seed(5)
    random.shuffle(data)
    Graph_Data = [GN.Graph(graph, max_node) for graph in data]
    labels = np.array([graph.y_value for graph in Graph_Data])
    labels = torch.FloatTensor(labels)
    # labels = load_graph_label()
    num_fold = 248
    folded_data = [data[i:i + num_fold] for i in range(0, len(Graph_Data), num_fold)]
    folded_labels = [Graph_Data[i:i + num_fold] for i in range(0, len(Graph_Data), num_fold)]
    for i in range(0, len(folded_labels)):
        folded_labels[i] = torch.FloatTensor(np.array([graph.y_value for graph in folded_labels[i]]))
    return folded_data,folded_labels,num_fold
#make 10 piece
folded_data,folded_labels,num_fold = load_data()
train_list,test_list,label_train,label_test,train_performance,test_performance= ([] for i in range(6)) #0-9


for i in range(0,1):
    bot = i*num_fold
    top = bot +len(folded_data[i])
    test_list.append(folded_data[i])
    label_test.append(folded_labels[i])
    temp = []
    for j in range(0,10):
        if j != i:
            temp.append(label for label in folded_labels[j])
    t = []
    for j in range(0,10):
        if j != i:
            t.append(folded_data[j])
    train_list.append(t)
    print(train_list)
    label_train.append(temp)