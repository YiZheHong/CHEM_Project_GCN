import statistics
import time
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import GN
import math
from pygcn.utils import load_dataset,get_idx_split,cross_valid,get_idx_split_noValid
from pygcn.models import GCN
from pygcn.eval import ThreeDEvaluator
from pygcn import run
from sklearn.metrics import mean_squared_error
from _csv import reader
def writetxt(best_trains,best_valids,best_tests,time,save_location):
    with open(save_location+'.txt', 'w') as f:
        f.write(f'Trains: {str(best_trains)}, Valids: {str(best_valids)}, Tests: {str(best_tests)}, Time: {str(abs(time))}')
def get_fold(size,fold=5):
    data = range(size)
    length = math.ceil((len(data) / fold))  # length of each fold
    folds = []
    for i in range(fold-1):
        folds += [data[i * length:(i + 1) * length]]
    folds += [data[(fold-1) * length:len(data)]]
    return folds
if __name__ == '__main__':
    torch.manual_seed(4)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    data_location = './data/CHEM_Data_2129.csv'
    A,F,L = load_dataset(data_location,shuffle = False)
    print(L[0])
    folds = get_fold(size = 2129)
    print(len(folds[4]))
    # split_idxs = cross_valid(folds)
    # split_idxs = get_idx_split(len(A))
    split_idxs = get_idx_split_noValid(len(A))

    loss_func = torch.nn.MSELoss()
    evaluation = ThreeDEvaluator()
    modleName = "GCN"
    start = time.time()
    save_location = f'./modelPerformance/2129Data/'
    graphConv_hids = [32]
    MLP_hids = [16]
    out_channelss = [128]
    for graphConv_hid in graphConv_hids:
        for MLP_hid in MLP_hids:
            for out_channels in out_channelss:
                best_tests = []
                best_valids = []
                best_trains = []
                for curr_fold in range(0,len(split_idxs)):
                    loc= f'{modleName}_{graphConv_hid}_{MLP_hid}_{out_channels}'
                    txt_location = save_location+f'BestRecord/{loc}'
                    split_idx = split_idxs[curr_fold]
                    train_index, valid_index, test_index = split_idx['train'], split_idx['valid'], split_idx['test']
                    model = GCN(node_feat=A[0].shape[0],
                                graph_feat = F[0].shape[1],
                                graphConv_hid=graphConv_hid,
                                MLP_hid = MLP_hid,out_channels = out_channels,
                                nclass=1,
                                dropout=0)
                    run3d = run()
                    run3d.run(curr_fold, modleName, device,A,F,L,train_index, valid_index, test_index, model, loss_func,
                              evaluation,save_location,loc, epochs=800,batch_size=12, vt_batch_size=12, lr=0.00255)
                    curr = start - time.time()
                    best_tests.append(round(run3d.best_test, 2))
                    best_trains.append(round(run3d.best_train, 2))
                    best_valids.append(round(run3d.best_valid, 2))

                    writetxt(best_trains, best_valids, best_tests, curr, txt_location)
                    print()
    # #
    # # #
