import torch
from opt import args
from utils import eva, target_distribution, calc_loss
from torch.optim import Adam
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans


def Train(model, data, adj, label, n_views,n_cluster,lr):
    with torch.no_grad():
        Z, A_hat, S, q = model(data, adj)

    # ----------------------initialize k-means----------------------
    features = S.data.cpu().detach()
    kmeans_ = KMeans(n_clusters=n_cluster, n_init=20)
    _ = kmeans_.fit_predict(features)
    model.cluster_layer.data = torch.tensor(kmeans_.cluster_centers_).to('cuda')

    optimizer = Adam(model.parameters(), lr=lr)
    
    for epoch in range(args.max_epoch):
        model.train()
        info_loss = torch.tensor(0.0, requires_grad=True).to('cuda')
        reconstruct_loss = torch.tensor(0.0, requires_grad=True).to('cuda')
        
        Z, A_hat, S, q= model(data, adj)
        tmp_q = q.data
        p = target_distribution(tmp_q)
    
        # --------------------------loss--------------------------
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        for v in range(n_views):
            reconstruct_loss += F.mse_loss(adj[v], A_hat[v])
            info_loss += calc_loss(S, Z[v],temperature=args.temperature)
            
        model_loss = reconstruct_loss + info_loss +  kl_loss
        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()
        
        # --------------------------eval--------------------------
        if epoch % 100 ==0:
            kmeans_ = KMeans(n_clusters=n_cluster, n_init=20).fit(S.data.cpu().numpy())
            y_pred = kmeans_.predict(S.data.cpu().numpy())
            eva(label, y_pred, epoch)
    
