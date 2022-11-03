import os
import torch
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import pandas as pd
from pygcn.utils import DataLoader

class run():
    r"""
    The base script for running different 3DGN methods.
    """

    def __init__(self):
        pass
    def record(self,save_location,loc):
        Lossloc = save_location+f'LossRecord/{loc}'+f'_fold{self.fold+1}'
        Resultloc = save_location+f'ResultRecord/{loc}'+f'_fold{self.fold+1}'

        loss_train_loc = Lossloc+'_train'
        loss_valid_loc = Lossloc+'_valid'
        loss_test_loc = Lossloc+'_test'
        result_train_loc = Resultloc + '_train'
        result_valid_loc = Resultloc + '_valid'
        result_test_loc = Resultloc + '_test'
        with open(loss_train_loc+"Loss.txt", 'w') as f:
            f.write(str(self.train_loss))
        
        with open(loss_test_loc+"Loss.txt", 'w') as f:
            f.write(str(self.test_loss))
        self.trainDf.to_csv(result_train_loc+"Result.csv",encoding='utf-8', index=False)
        # with open(loss_valid_loc+"Loss.txt", 'w') as f:
        #     f.write(str(self.valid_loss))
        # self.validDf.to_csv(result_valid_loc+"Result.csv",encoding='utf-8', index=False)
        self.testDf.to_csv(result_test_loc+"Result.csv",encoding='utf-8', index=False)


    def run(self, curr_fold, modelName, device,A,F,L,train_index, valid_index, test_index, model, loss_func,
                  evaluation,save_location,loc, epochs=20,batch_size=12, vt_batch_size=12, lr=0.00055,weight_decay=0):
        self.fold = curr_fold
        self.modelName = modelName
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.validDf = {}
        self.testDf = {}

        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loader = DataLoader(A,F,L,train_index, valid_index, test_index,batch_size)
        train_loader = loader['train_loader']
        # valid_loader = loader['valid_loader']
        test_loader = loader['test_loader']


        self.best_valid = float('inf')
        self.best_test = float('inf')

        for epoch in range(1, epochs + 1):
            if epoch % 10 == 0:
                print("\n=====Epoch {}".format(epoch), flush=True)

            # print('\nTraining...', flush=True)
            train_mae,trdf = self.train(model, train_loader,optimizer, loss_func, device,evaluation)
            self.train_loss.append(train_mae)

            # print('\n\nEvaluating...', flush=True)
            # valid_mae,vDf = self.val(model, valid_loader,evaluation, device)
            # self.valid_loss.append(valid_mae)

            # print('\n\nTesting...', flush=True)
            test_mae,tDf = self.val(model, test_loader, evaluation, device)
            self.test_loss.append(test_mae)

            # print()
            if epoch % 10 == 0:
                # print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})
                print({'Train': train_mae, 'Test': test_mae})

            # if valid_mae < self.best_valid:
            if test_mae < self.best_test:
                # self.best_valid = valid_mae
                self.best_test = test_mae
                self.best_train = train_mae
                # self.validDf = vDf
                self.testDf = tDf
                self.trainDf = trdf

        self.record(save_location,loc)
        print(f'Best validation RMSE so far: {self.best_valid}')
        print(f'Test RMSE when got best validation result: {self.best_test}')


    def train(self, model,train_loader, optimizer,loss_func, device,evaluation):
        r"""
        The script for training.

        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function): The used loss funtion for training.
            device (torch.device): The device where the model is deployed.
        :rtype: Traning loss. ( :obj:`mae`)

        """
        model.train()
        trues = []
        preds = []
        loss_accum = 0
        for aBatch,fBatch,yBatch in zip(train_loader['A'],train_loader['F'],train_loader['Y']):
            optimizer.zero_grad()
            MSE_total = 0
            for data in range(len(aBatch)):
                out = model(aBatch[data],aBatch[data],fBatch[data])
                MSE_loss = loss_func(out, yBatch[data])
                MSE_total+=MSE_loss
                trues.append(round(yBatch[data].item(), 2))
                preds.append(round(out.item(), 2))
            MSE_total /= len(aBatch)
            # RMSE_loss = torch.sqrt(MSE_total)
            MSE_total.backward()
            # RMSE_loss.backward()
            optimizer.step()
            loss_accum += MSE_total
        dic = {"y_true": trues, "y_pred": preds}
        df = pd.DataFrame.from_dict(dic)
        return np.sqrt(loss_accum.item() / len(train_loader['A'])),df

    def val(self, model, data_loader,evaluation, device):
        model.eval()

        preds = []
        targets = []

        for aBatch,fBatch,yBatch in zip(data_loader['A'],data_loader['F'],data_loader['Y']):
            for data in range(len(aBatch)):
                out = model(aBatch[data], aBatch[data],fBatch[data])
                preds.append(out)
                targets.append(yBatch[data])
        input_dict = {"y_true": torch.Tensor(targets), "y_pred": torch.Tensor(preds)}
        trues = []
        preds = []
        for i in range(len(input_dict['y_true'])):
            trues.append(round(input_dict['y_true'][i].item(), 2))
            preds.append(round(input_dict['y_pred'][i].item(), 2))
        dic = {"y_true": trues, "y_pred": preds}
        df = pd.DataFrame.from_dict(dic)

        return evaluation.eval(input_dict)['rmse'],df