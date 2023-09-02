import torch
import numpy as np
from torch.utils.data import Dataset
import opt
import scipy.io as sio


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), torch.from_numpy(np.array(idx))
               
               
def load_data(data_name):
    
    device = torch.device("cuda" if opt.args.cuda else "cpu")
    if data_name == 'acm':
        n_views = 2
        n_cluster = 3
        data_ = sio.loadmat('./data/normalized_ACM.mat')
        Y = data_['y'][:, 0]
        A_hat = data_['A_hat'].T
        X = data_['X']
        A = []
        for v in range(n_views):
            A.append(A_hat[v][0].astype(np.float32))
        A = torch.tensor(A).type(torch.FloatTensor).to(device)
        
    return X, A, Y, n_cluster, n_views