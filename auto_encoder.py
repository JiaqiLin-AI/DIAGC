import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module,Parameter
import numpy as np
import opt


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        if opt.args.name == "acm":
            self.w = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.w = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.w)
        
    def forward(self, features, adj, active=False):
        if active:
            if opt.args.name == "acm":
                support = self.act(torch.mm(features, self.w))
            else:
                support = self.act(F.linear(features, self.w))  # add bias
        else:
            if opt.args.name == "acm":
                support = torch.mm(features, self.w)
            else:
                support = F.linear(features, self.w)  # add bias
        output = torch.mm(adj, support)
        return output


class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, hidden_dim, x_dim):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(x_dim, gae_n_enc_1)  
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2) 
        self.gnn_3 = GNNLayer(gae_n_enc_2, hidden_dim) 

    def forward(self, x, adj):
        z = self.gnn_1(x, adj, active=True)
        z = self.gnn_2(z, adj, active=True)
        z_igae = self.gnn_3(z, adj, active=False)
        return z_igae


class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, hidden_dim):
        super(IGAE_decoder, self).__init__()
        self.decoder_1 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.decoder_2 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.decoder_3 = GNNLayer(gae_n_dec_3, hidden_dim)

    def forward(self, x, adj):
        z = self.decoder_1(x, adj, active=True)
        z = self.decoder_2(z, adj, active=True)
        z_hat = self.decoder_3(z, adj, active=False)
        return z_hat

class Inner_product_decoder(nn.Module):
    def __init__(self, input_dim):
        super(Inner_product_decoder, self).__init__()
        self.weight = torch.tensor(np.eye(input_dim),requires_grad=True,dtype=torch.float32).to('cuda') 
        self.act = torch.nn.Sigmoid()
    def forward(self,z):
        tmp = torch.mm(z,self.weight)
        a = torch.mm(tmp, torch.transpose(input=z, dim0=0, dim1=1))
        output = self.act(a)
        return output