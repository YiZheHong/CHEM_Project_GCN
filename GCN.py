import statistics
import time
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import GN
from pygcn.utils import load_dataset,get_idx_split
from pygcn.models import GCN
from pygcn.eval import ThreeDEvaluator
from pygcn import run
from sklearn.metrics import mean_squared_error
from _csv import reader
def writetxt(best_trains,best_valids,best_tests,time,save_location):
    with open(save_location, 'w') as f:
        f.write(f'Trains: {str(best_trains)}, Valids: {str(best_valids)}, Tests: {str(best_tests)}, Time: {str(abs(time))}')

if __name__ == '__main__':
    best_tests = []
    best_valids = []
    best_trains = []
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    data_location = './data/CHEM_Data_2129.csv'
    A,F,L = load_dataset(data_location)
    split_idxs = get_idx_split(len(A))
    loss_func = torch.nn.MSELoss()
    evaluation = ThreeDEvaluator()
    modleName = "GCN"
    save_location = f'./modelPerformance/2129Data/{modleName}.txt'
    start = time.time()

    for curr_fold in range(len(split_idxs)):
        split_idx = split_idxs[curr_fold]
        train_index, valid_index, test_index = split_idx['train'], split_idx['valid'], split_idx['test']
        model = GCN(nfeat=F[0].shape[1],
                    nhid=16,
                    nclass=1,
                    dropout=0.5)
        run3d = run()
        run3d.run(curr_fold, modleName, device,A,F,L,train_index, valid_index, test_index, model, loss_func,
                  evaluation, epochs=100,batch_size=12, vt_batch_size=12, lr=0.00755)
        curr = start - time.time()
        best_tests.append(round(run3d.best_test, 2))
        best_trains.append(round(run3d.best_train, 2))
        best_valids.append(round(run3d.best_valid, 2))
        writetxt(best_trains, best_valids, best_tests, curr, save_location)


