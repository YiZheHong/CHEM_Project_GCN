

import math
import statistics
import time
import argparse
from _csv import reader
import random

import numpy as np

import torch
import torch.optim as optim

import GN
from utils import load_graph_data, load_graph_label
import models
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
data = []
global max_node
max_node = 7
std = []
with open('../data/GCN_data.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    j = 0
    for row in csv_reader:
        if j != 0:
            data.append(row)
            std.append(float(row[2]))
            if len(row) - 3 > max_node:
                max_node = len(row) - 3
        j += 1

random.seed(5)
std = statistics.pstdev(std)
print('std',std)
random.shuffle(data)
labels = load_graph_label()
num_fold = 205
folded_data = [data[i:i + num_fold] for i in range(0, len(data), num_fold)]
#make 10 piece

train_list= [] #[1-9]....9 lists
test_list= [] #0-9
label_train = [] #[1-9]
label_test = [] #[0]
train_performance = []
test_performance = []

for i in range(0,10):
    bot = i*num_fold
    top = bot +len(folded_data[i])
    test_list.append(folded_data[i])
    label_test.append(labels[bot:top])
    temp = []
    for index in range(0,2048):
        if index < bot or index >=top:
            y_value = round(float(labels[index]),2)
            temp.append(y_value)
    t = []
    for j in range(0,10):
        if j != i:
            t.append(folded_data[j])
    train_list.append(t)
    label_train.append(temp)

n_val= 10
for k in range(0,10):
    train_i = train_list[k]
    train_graph = []
    for i in range(0,len(train_i)):
        for j in range(0,len(train_i[i])):
            train_graph.append(train_i[i][j])

    train_batch = [GN.Graph(graph,max_node) for graph in train_graph]
    test_batch = [GN.Graph(graph,max_node) for graph in test_list[k]]

    adj, features = load_graph_data(train_batch[0])

    A_train= []
    F_train = []
    A_test= []
    F_test = []
    L_train = label_train[k]
    L_test = label_test[k]
    model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=1,
                    dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)


    for graph, i in zip(train_batch, range(0,len(train_batch))):

        adj, features = load_graph_data(graph)
        A_train.append(adj)
        F_train.append(features)

    for graph, i in zip(test_batch, range(0,len(test_batch))):

        adj, features = load_graph_data(graph)
        A_test.append(adj)
        F_test.append(features)


    # # Model and optimizer
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    def train(epoch):
        sum = 0
        t = time.time()
        model.train() # dropout batch norm 激活

        # 66-67 forloop
        #RMSE_train = 0
        for i in range(0, len(A_train)):
            optimizer.zero_grad()  # 把梯度清0

            output = model(F_train[i], A_train[i])
            if train_batch[i].y_value != L_train[i]:
                print(k,i,'false')
            RMSE_train = (output[0] - L_train[i])**2
            sum+=RMSE_train
            RMSE_train.backward()  # 计算梯度
            optimizer.step()  # 梯度往前走一部



        sum = math.sqrt(sum/len(A_train))



        print('Epoch: {:04d}'.format(epoch+1),
              'RMSE_train: {:.4f}'.format(sum),
              'time: {:.4f}s'.format(time.time() - t))
        return sum


    def test():
        model.eval()
        RMSE_test = 0
        for i in range(0, len(A_test)):
            output = model(F_test[i], A_test[i])
            if test_batch[i].y_value != L_test[i]:
                print(k,i,'false')
            RMSE_test += (output[0] - L_test[i]) ** 2 / len(A_test)
        RMSE_test = torch.sqrt(RMSE_test)
        test_performance.append(RMSE_test)

        print("Test set results:",
              "RMSE= {:.4f}".format(RMSE_test.item()))
    # # # Train model
    t_total = time.time()
    RMSE_T = 0
    for epoch in range(args.epochs):
    # for epoch in range(2):
        RMSE_T=train(epoch)
        # print("\n\n\n\n\n\n")

    train_performance.append(RMSE_T)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    #
    # # # Testing
    test()

print("\n\n\n\n\n\n")
# print(type(train_performance[0]))
Tr_p = [i for i in train_performance]
Te_p = [i.item() for i in test_performance]
#
std_train = 0
std_test = 0
mean_train = 0
mean_test = 0


std_train = statistics.pstdev(Tr_p)
std_test = statistics.pstdev(Te_p)
mean_test = statistics.mean(Te_p)
mean_train = statistics.mean(Tr_p)

print("RMSEs of Train",Tr_p)
print("Mean of Train", mean_train)
print("STD of Train",std_train)
print("\n\n")
print("RMSEs of Test", Te_p)
print("Mean of Test", mean_test)
print("STD of Test",std_test)