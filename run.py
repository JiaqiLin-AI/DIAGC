import warnings
import torch
from opt import args
from DIAGC_model import DIAGC
from train import Train
from sklearn.decomposition import PCA
from data_loader import LoadDataset, load_data
warnings.filterwarnings('ignore')
print("\nuse cuda: {}".format(args.cuda))


def main(X, A, Y, n_cluster, n_views):
    x1 = PCA(n_components=args.n_components).fit_transform(X)
    dataset = LoadDataset(x1)
    data = torch.Tensor(dataset.x).type(torch.FloatTensor).to('cuda')
    
    model = DIAGC(x_dim=data.shape[1], n_input=args.n_components,
                  n_views=n_views, n_clusters=n_cluster).to('cuda')
    
    print('data_name = {}, lr = {}\n'.format(args.name, args.lr))
    print('#---------------------------training---------------------------')
    
    Train(model, data, A, Y, n_views=n_views, n_cluster=n_cluster, lr=args.lr)
    
    print('#---------------------------finished---------------------------')
    

if __name__ == '__main__':
    X, A, Y, n_cluster, n_views = load_data(args.name)
    main(X, A, Y, n_cluster, n_views)


