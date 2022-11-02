import numpy as np
import scipy.sparse as sp
from csv import reader
import GN
import random
import torch



def load_graph_data(graph):
    features = [graph.charge,graph.num_of_chem,graph.Dis,graph.num_edges / graph.num_of_chem,graph.sum_dist / graph.num_of_chem,graph.Vib]
    features = sp.csr_matrix(features,dtype=np.float32)
    adj = torch.from_numpy(graph.A_sc)
    # features = normalize(features)
    # adj = normalize(adj + torch.eye(adj.shape[0]))
    features = torch.FloatTensor(np.array(features.todense()))
    # adj = torch.from_numpy(adj)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features

def DataLoader(A,F,L,train_index, valid_index, test_index,train_batchSize):
    loader = {}
    for indexs,name in zip([train_index,valid_index,test_index],['train_loader','valid_loader','test_loader']):
        a_batches = []
        f_batches = []
        y_true_batches = []
        aBatch = []
        fBatch = []
        yBatch = []
        print(name,len(indexs))
        for i in range(len(indexs)):
            index = indexs[i]
            aBatch.append(A[index])
            fBatch.append(F[index])
            yBatch.append(L[index])
            if len(aBatch) == train_batchSize:
                a_batches.append(aBatch)
                f_batches.append(fBatch)
                y_true_batches.append(yBatch)
                aBatch = []
                fBatch = []
                yBatch = []
            elif i == len(indexs)-1:
                a_batches.append(aBatch)
                f_batches.append(fBatch)
                y_true_batches.append(yBatch)
                aBatch = []
                fBatch = []
                yBatch = []
        loader[name]={'A':a_batches,'F':f_batches,'Y':y_true_batches}
    return loader

def load_dataset(data_location,shuffle = False):
    data = []
    A = []
    F = []
    max_node = 7
    with open(data_location, 'r') as read_obj:
        csv_reader = reader(read_obj)
        j = 0
        for row in csv_reader:
            if j != 0:
                data.append(row)
            j += 1
    Graph_Data = [GN.Graph(graph, max_node) for graph in data]
    if shuffle:
        random.Random(4).shuffle(Graph_Data)
    labels = [torch.FloatTensor([graph.y_value]) for graph in Graph_Data]
    for graph in Graph_Data:
        adj, features = load_graph_data(graph)
        A.append(adj)
        F.append(features)
    return A,F,labels

def cross_valid(folds):
    split_idxs = []
    last = False
    for i in range(len(folds)):
        train = []
        valid = folds[i]
        if i+1 < len(folds):
            test = folds[i+1]
        else:
            last = True
            test = folds[0]
        for j in range(len(folds)):
            if last != True:
                if j != i and j != i + 1:
                    for idx in folds[j]:
                        train.append(idx)
            else:
                if j != i and j != 0:
                    for idx in folds[j]:
                        train.append(idx)
        dic = {'train':train,'valid':valid,'test':test}
        split_idxs.append(dic)
    return split_idxs
def formValid():
        d = {'2': 24, '3': 91, '4': 260, '5': 316, '6': 627, '7': 477}
        nums = [3, 17, 47, 60, 117, 90]
        bot = 0
        top = 0
        valid = []
        for key, num in zip(d, nums):
            top += d[key]
            l = random.Random(5).sample(range(bot, top), k=num)
            valid.extend(l)
            bot = top
        print(valid[:5])
        return valid
def get_idx_split(data_size,shuffle = True):
        split_dicts = []
        val_idx = formValid()
        train_idx = [i for i in range(1795) if i not in val_idx]
        test_idx = range(1795,data_size)
        if shuffle:
            random.Random(4).shuffle(train_idx)
        print(len(train_idx),len(val_idx))
        for val in range(0, 1):
            split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
            split_dicts.append(split_dict)
        return split_dicts


def get_idx_split_noValid(data_size,shuffle = True):
    split_dicts = []
    train_idx = [i for i in range(1795)]
    test_idx = range(1795, data_size)
    if shuffle:
        random.Random(4).shuffle(train_idx)
    for val in range(0, 1):
        split_dict = {'train': train_idx, 'valid': [], 'test': test_idx}
        split_dicts.append(split_dict)
    return split_dicts

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
