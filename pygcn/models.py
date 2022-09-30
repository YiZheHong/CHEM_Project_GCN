import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn

from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid) #nhid = H^(nhid)
        self.gc2 = GraphConvolution(nhid, 64)
        self.linear3 = nn.Linear(64,32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, nclass)
        self.dropout = 0

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)

        x,_= torch.max(x,dim=0)
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x

