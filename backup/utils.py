import numpy as np
import scipy.sparse as sp
from csv import reader
import GN
import random
import torch

global batch
batch = []

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_graph_data(graph):
    features = [[graph.charge],[graph.num_of_chem],[graph.M],[graph.Vib],[graph.Dis],[graph.num_edges / graph.num_of_chem],[graph.sum_dist / graph.num_of_chem]]
    # features.append([graph.charge])

    features = sp.csr_matrix(features,dtype=np.float32)
    adj = torch.from_numpy(graph.A_sc)
    # features = normalize(features)

    adj = normalize(adj + torch.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    adj = torch.from_numpy(adj)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)


    return adj, features

def load_graph_label(path="./CHEM_Data.csv", seed=5):
    print('loading data from', path)
    data = []
    max_node = 0
    with open(path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)

        # Iterate over each row in the csv using reader object
        j = 0
        for row in csv_reader:
            if j != 0:
                data.append(row)
                if len(row) - 3 > max_node:
                    max_node = len(row) - 3
            j += 1
    random.seed(seed)
    random.shuffle(data)

    l = [GN.Graph(graph, max_node) for graph in data]
    labels = np.array([graph.y_value for graph in l])
    labels = torch.FloatTensor(labels)
    return labels




def normalize(mx):
    """Row-normalize sparse matrix"""

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
