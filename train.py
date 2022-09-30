
from sklearn.metrics import mean_squared_error

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
from utils import load_graph_data
# from data.models import GCN
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
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
train_performance, test_performance = [],[]
start = time.time()
# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
data = []
max_node = 7
with open('data/CHEM_Data_2129.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    j = 0
    for row in csv_reader:
        if j != 0:
            data.append(row)
        j += 1

# random.seed(4)
# random.shuffle(data)
Graph_Data = [GN.Graph(graph, max_node) for graph in data]
labels = np.array([graph.y_value for graph in Graph_Data])
def load_data(Graph_Data,labels):

    labels = torch.FloatTensor(labels)
    # labels = load_graph_label()
    num_fold = 199
    folded_data = [Graph_Data[i:i + num_fold] for i in range(0, len(Graph_Data), num_fold)]
    folded_labels = [Graph_Data[i:i + num_fold] for i in range(0, len(Graph_Data), num_fold)]

    for i in range(0, len(folded_labels)):
        folded_labels[i] = torch.FloatTensor(np.array([graph.y_value for graph in folded_labels[i]]))

    return folded_data,folded_labels,num_fold
# make 10 piece
# folded_data,folded_labels,num_fold = load_data(Graph_Data,labels)
# train_list,test_list,label_train,label_test,train_performance,test_performance= ([] for i in range(6)) #0-9


# for i in range(9,10):
#     test_list.append(folded_data[i])
#     label_test.append(folded_labels[i])
#     temp = []
#     for j in range(0,10):
#         if j != i:
#             for label in folded_labels[j]:
#                 temp.append(label)
#     t = []
#     for j in range(0,10):
#         if j != i:
#             t.append(folded_data[j])
#     train_list.append(t)
#     label_train.append(temp)

n_val= 10
for k in range(0,1):
    true_test = []
    pred_test = []
    # print(len(train_list[0]))
    # train_i = train_list[k]
    # train_batch = []
    # for i in range(0,len(train_i)):
    #     for j in range(0,len(train_i[i])):
    #         train_batch.append(train_i[i][j])
    #
    # test_batch = test_list[k].copy()
    train_batch = []
    test_batch = []
    label_train = []
    label_test = []


    for i in range(1795):
        train_batch.append(Graph_Data[i])
        label_train.append(labels[i])
    for i in range(1795,2129):
        test_batch.append(Graph_Data[i])
        label_test.append(labels[i])
    random.Random(4).shuffle(train_batch)
    random.Random(4).shuffle(label_train)

    # label_train = torch.FloatTensor(np.array([label_train]))
    # label_test = torch.FloatTensor(np.array([label_test]))
    adj, features = load_graph_data(train_batch[0])
    A_train= []
    F_train = []
    A_test= []
    F_test = []
    L_train = label_train
    L_test = label_test
    # print(features.shape[1])
    model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=1,
                    dropout=args.dropout)
    optimizer = optim.AdamW(model.parameters(),
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
        true_train = []
        pred_train = []
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
            true_train.append(L_train[i])
            pred_train.append(output[0].item())
            RMSE_train.backward()  # 计算梯度
            optimizer.step()  # 梯度往前走一部
        # print(pred_train)
        # print(true_train)
        rmse = mean_squared_error(true_train, pred_train, squared=False)
        print('Epoch: {:04d}'.format(epoch+1),
              'RMSE_train: {:.4f}'.format(rmse),
              'time: {:.4f}s'.format(time.time() - t))
        return rmse


    def test():
        true_test = []
        pred_test = []
        model.eval()
        for i in range(0, len(A_test)):
            output = model(F_test[i], A_test[i])
            true_test.append(L_test[i])
            pred_test.append(output[0].item())
        # print(pred_test)
        rmse = mean_squared_error(true_test, pred_test, squared=False)

        test_performance.append(rmse)

        print("Test set results:",
              "RMSE= {:.4f}".format(rmse))
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
def print_stat(train_performance,test_performance):
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

print_stat(train_performance,test_performance)
end = time.time()
print('time:',end-start)