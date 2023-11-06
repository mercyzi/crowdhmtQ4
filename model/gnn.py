

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch.nn import Dropout,Tanh,Sigmoid
from torch_geometric.nn import GCNConv, GATConv, Linear
from torch_geometric.utils import negative_sampling


class MyGAT(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim):
        super(MyGAT, self).__init__()
        self.GAT1 = GATConv(hidden_channels, hidden_channels, edge_dim)
        self.dropout1 = Dropout(p=0.1)
        self.linear1 = Linear(edge_dim * hidden_channels, hidden_channels)
        self.GAT2 = GATConv(hidden_channels, hidden_channels, edge_dim)
        # self.dropout2 = Dropout(p=0.1)

        # self.linear2 = Linear(edge_dim * hidden_channels, hidden_channels)
        self.GAT3 = GATConv(hidden_channels, 1, edge_dim)
        # self.linear3 = Linear(edge_dim * hidden_channels, hidden_channels)
        self.GAT4 = GATConv(edge_dim, 1, edge_dim)
        # self.linear4 = Linear(edge_dim * hidden_channels, hidden_channels)
        # self.GAT5 = GATConv(edge_dim, 1, edge_dim)
        self.linear5 = Linear(edge_dim * hidden_channels, edge_dim)
 

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.GAT1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x = self.dropout1(self.linear1(x))
        # print(x.shape)
        x = F.relu(self.GAT2(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x = self.linear5(x)
        # x = F.relu(self.GAT3(x=x, edge_index=edge_index, edge_attr=edge_attr))
        # x = F.relu(self.GAT4(x=x, edge_index=edge_index, edge_attr=edge_attr))
        # x = F.relu(self.GAT5(x=x, edge_index=edge_index, edge_attr=edge_attr))
        
        
        return x

# class Classifier(torch.nn.Module):
#     def __init__(self, num_edge_features):
#         super(Classifier, self).__init__()
#         self.edge_w = Linear(3 * num_edge_features, 1) 
#         self.classifier1 = Linear(num_edge_features, 2)  
#         self.classifier2 = Linear(num_edge_features, 2)

#     def forward(self, x, edge_label_index, edge_label_attr):
#         # Convert node embeddings to edge-level representations:
#         edge_feat_user = x[edge_label_index[0]]
#         edge_feat_movie = x[edge_label_index[1]]
#         output = self.edge_w(torch.cat((edge_feat_user, edge_label_attr, edge_feat_movie), dim=-1))
#         return output.squeeze()  

class Classifier(torch.nn.Module):
    def __init__(self, num_edge_features):
        super(Classifier, self).__init__()
        self.edge_w = Linear(2 * num_edge_features, num_edge_features) 
        # self.linear = Linear(num_edge_features, 128)
        self.activation = Sigmoid()
    def forward(self, x, edge_label_index, edge_label_attr, topic):
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x[edge_label_index[0]]
        edge_feat_movie = x[edge_label_index[1]]
        
        edge_feat_user = self.edge_w(torch.cat((edge_feat_user, edge_label_attr), dim=-1))
        # edge_feat_movie = self.linear(edge_feat_movie)
        # Apply dot-product to get a prediction per supervision edge:
        
        return self.activation((edge_feat_user * edge_feat_movie).sum(dim=-1))

class Model(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_edge_features):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users:
        
        self.user_emb = torch.nn.Embedding(num_nodes, hidden_channels)

        # Instantiate GAT:
        self.gnn = MyGAT(hidden_channels, num_edge_features)


        self.classifier = Classifier(num_edge_features)

    def forward(self, data):
        
        x= self.gnn(self.user_emb(data.x), data.edge_index, data.edge_attr)

        pred = self.classifier(
            x,
            data.edge_label_index, data.edge_label_attr, data.topic
        )
        return pred