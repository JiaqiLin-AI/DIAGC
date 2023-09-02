import torch
from torch import nn
from auto_encoder import IGAE_encoder, IGAE_decoder
from opt import args

class DIAGC(nn.Module):
    def __init__(self, x_dim, n_views, n_input, n_clusters):
        super(DIAGC, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=args.gae_n_enc_1,
            gae_n_enc_2=args.gae_n_enc_2,
            hidden_dim=args.gae_n_enc_3,
            x_dim=x_dim
        )
        self.decoder = IGAE_decoder(
            gae_n_dec_1=args.gae_n_enc_3,
            gae_n_dec_2=args.gae_n_enc_2,
            gae_n_dec_3=args.gae_n_enc_1,
            hidden_dim=n_input
        )
        self.n_views = n_views
        self.S_net = MLP(in_dims=n_views * args.gae_n_enc_3, hidden_dim=args.gae_n_enc_3).to('cuda') 

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, args.gae_n_enc_3), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = 1.0
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        A_hat = []
        Z = []
        en_a_hat = []
        de_a_hat =[]
        for v in range(self.n_views):
            Z.append(self.encoder(x, adj[v]))
            en_a_hat.append(self.s(torch.mm(Z[v], Z[v].t())))

        S = self.S_net(torch.cat(Z, 1))
        for v in range(self.n_views):
            tmp_A = self.decoder(Z[v], adj[v])
            de_a_hat.append(self.s(torch.mm(tmp_A, tmp_A.t())))
            A_hat.append(de_a_hat[v]+en_a_hat[v])
            
        q = 1.0 / (1.0 + torch.sum(torch.pow((S).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return Z, A_hat, S, q


class MLP(torch.nn.Module): 
    def __init__(self, in_dims, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_dims, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc3 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim,hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim,hidden_dim)
        self.act = nn.Sigmoid()
        
    def forward(self, din):
        out = self.act(self.fc1(din))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = self.act(self.fc4(out))
        out = self.fc5(out)
        return out


