import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn

from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, node_feat,graph_feat, graphConv_hid,MLP_hid,out_channels,nclass, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(node_feat, graphConv_hid) #nhid = H^(nhid)
        # self.gc3 = GraphConvolution(int(graphConv_hid), int(graphConv_hid/2))
        self.gc2 = GraphConvolution(int(graphConv_hid), out_channels)
        self.linear1 = nn.Linear(graph_feat,MLP_hid)
        self.linear2 = nn.Linear(MLP_hid, out_channels)

        self.linear3  = nn.Linear(out_channels*2, out_channels)
        self.linear4 = nn.Linear(int(out_channels), int(out_channels/2))
        self.linear5 = nn.Linear(int(out_channels/2), int(out_channels/4))
        self.linear6 = nn.Linear(int(out_channels/4), int(out_channels/8))
        self.linear7 = nn.Linear(int(out_channels/8), nclass)

    def forward(self, x, adj,graph_x):
        x = F.relu(self.gc1(x, adj))
        # x = F.relu(self.gc3(x,adj))
        x = self.gc2(x, adj)
        x,_= torch.max(x,dim=0)
        graph_x = torch.reshape(graph_x,(-1,))
        graph_x = F.relu(self.linear1(graph_x))
        graph_x = F.relu(self.linear2(graph_x))
        x = torch.cat([x,graph_x],dim=-1)
        x = F.relu(self.linear3(x))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))

        x = F.relu(self.linear6(x))
        x = self.linear7(x)

        return x

