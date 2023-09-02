import argparse
import pandas as pd
import os
from utils import setup_seed
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
setup_seed()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


parser = argparse.ArgumentParser(description='DIAGC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='acm')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--max_epoch', type=int, default=1001)
parser.add_argument('--temperature', type=float, default=0.3)

parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=512)
parser.add_argument('--gae_n_enc_3', type=int, default=700)
parser.add_argument('--lr', type=float, default=1e-4,help=' Learning rate.')


args = parser.parse_args()