#!/usr/bin/env python
# coding: utf-8

import os
import joblib
from matplotlib import pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import random
import logging
import numpy as np
import pandas as pd
from itertools import chain
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

setup_seed(0)

train_id = [1, 2, 3]
val_id = [4]
test_id = [5]

battery_data_path = "./data/"
result_path_prefix = "./results/"
features = ["V", "VT", "VP", "VS", "VTP", "VTS", "VPS", "VTPS"]

# Hyperparameters
weight_decay = 0
save_step = 1
L = 10
num_workers = 0

out_channels = 1
hidden_size = 128
num_layers = 3
patience = 30
lr = 0.0001
bs = 8
epochs = 50
if len(sys.argv) > 1:
    hidden_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_layers = int(sys.argv[2])
result_path_prefix = os.path.join(result_path_prefix, 'combined_feature')

log = None
def create_logger(exp_folder, file_name, log_file_only=False):
    handlers = [] if log_file_only else [logging.StreamHandler(sys.stdout)]
    if file_name != '':
        log_path = os.path.join(exp_folder, file_name)
        os.makedirs(os.path.split(log_path)[0], exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode='w'))
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', handlers=handlers)
    global log
    log = logging.getLogger()


def destroy_logger():
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)

def load_obj(path):
    with open(path, 'rb') as f:
        return joblib.load(f)
def save_obj(obj, path):
    with open(path, 'wb') as f:
        joblib.dump(obj, f)
class CNN_LSTM(torch.nn.Module):
    def __init__(self, L, input_size, hidden_size, out_channels, num_layers):
        super().__init__()
        self.L = L
        self.input_size = input_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn1d = torch.nn.Conv1d(in_channels=self.input_size, out_channels=self.out_channels, kernel_size=1)
        self.norm = torch.nn.BatchNorm1d(self.out_channels)
        self.ac = torch.nn.Sigmoid()
        self.lstm = torch.nn.LSTM(self.out_channels, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, input_seq):
        input_seq = input_seq.permute(0, 2, 1)
        input_lstm = self.cnn1d(input_seq)
        input_lstm = self.norm(input_lstm)
        input_lstm = self.ac(input_lstm)
        input_lstm = input_lstm.permute(0, 2, 1)
        output, _ = self.lstm(input_lstm)
        output = self.fc(output)
        output = output[:, -1, :]
        return output

class MyDataset(Dataset):
    def __init__(self, battery_list, feature):
        all_fea, all_lbl = [], []
        for battery_path in battery_list:
            A = load_obj(battery_path)
            fea = A['fea']
            lbl = A['lbl']
            if feature == "V":
                data = fea[:, :2]
            if feature == "VT":
                data = fea[:, :3]
            if feature == "VP":
                temp = fea[:, :2]
                P = fea[:, 3]
                P = np.expand_dims(P, axis=-1)
                data = np.concatenate((temp, P), axis=-1)
            if feature == "VS":
                temp = fea[:, :2]
                S = fea[:, -1]
                S = np.expand_dims(S, axis=-1)
                data = np.concatenate((temp, S), axis=-1)
            if feature =="VTP":
                data = fea[:, :4]
            if feature =="VTS":
                temp = fea[:, :3]
                S = fea[:, -1]
                S = np.expand_dims(S, axis=-1)
                data = np.concatenate((temp, S), axis=-1)
            if feature =="VPS":
                temp = fea[:, :2]
                PS = fea[:, 3:]
                data = np.concatenate((temp, PS), axis=-1)
            if feature == "VTPS":
                data = fea[:, :]
            for i in range(len(data) - L):
                seq = data[i:i + L, :]
                seq = np.expand_dims(seq, axis=0)
                label = lbl[i + L, 0]
                all_fea.append(seq)
                all_lbl.append(label)
        all_fea = np.vstack(all_fea)
        all_lbl = np.vstack(all_lbl)
        self.dimension = all_fea.shape[-1]
        self.fea = all_fea.astype(np.float32)
        self.lbl = all_lbl.astype(np.float32)
    def __len__(self):
        return len(self.fea)

    def __getitem__(self, item):
        return self.fea[item], self.lbl[item]

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, epoch):
        if self.verbose:
            log.info("Save current best model @epoch {}".format(epoch))
        torch.save(model.state_dict(), path+'/'+ 'model.pt')
        self.val_loss_min = val_loss


class Trainer():
    def __init__(self, feature, epochs, lr, weight_decay, save_path, patience):


        if feature == 'VTP':
            self.epochs = epochs
            self.lr = 0.00079
        elif feature == 'VTS':
            self.epochs = 60
            self.lr = lr
        elif feature == 'VPS':
            self.epochs = epochs
            self.lr = lr
        elif feature == 'VTPS':
            self.epochs = 100
            self.lr = 0.000059
        else:
            self.epochs = epochs
            self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay
        self.save_path = save_path

        self.verbose = True

    def train(self, net, loader_train, loader_val):

        early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose)

        train_mae_list, train_rmse_list = [], []
        loss_function = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=0)
        for epoch in range(self.epochs):
            totla_train_loss = []
            train_pred, train_label = [], []
            for train_x, train_y in loader_train:
                train_x = train_x.to(device)
                train_y = train_y.to(device)
                output = net(train_x)
                train_loss = loss_function(output, train_y)
                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()
                x = list(chain.from_iterable(output.data.tolist()))
                train_pred.extend(x)
                y = list(chain.from_iterable(train_y.data.tolist()))
                train_label.extend(y)
                totla_train_loss.append(train_loss.item())
            train_mae = mean_absolute_error(train_label, train_pred)
            train_rmse = np.sqrt(mean_squared_error(train_label, train_pred))
            train_mae_list.append(train_mae)
            train_rmse_list.append(train_rmse)
            # val
            net.eval()
            val_loss = self.val(net, loader_val)
            if (epoch + 1) % save_step == 0:
                log.info("Epoch: {:<3d}  | MAE:{:<6.6f} | RMSE:{:<6.6f} | Train_loss:{:<6.6f} | Val_loss:{:<6.6f}".format(epoch, train_mae, train_rmse, np.mean(np.array(totla_train_loss)), val_loss))
            early_stopping(val_loss, net, self.save_path, epoch)
            net.train()
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def val(self, net, loader):
        val_label, val_pred = [], []
        total_loss = []
        for val_x, val_y in loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            with torch.no_grad():
                pre = net(val_x)
            t_pre = list(chain.from_iterable(pre.data.tolist()))
            val_pred.extend(t_pre)
            val_y = list(chain.from_iterable(val_y.data.tolist()))
            val_label.extend(val_y)
            loss = mean_squared_error(val_label, val_pred)
            total_loss.append(loss)
        val_loss = mean_squared_error(val_label, val_pred)
        return val_loss

    def test(self, net, loader, name):
        test_label, test_pred = [], []
        for test_x, test_y in loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            with torch.no_grad():
                pre = net(test_x)
            t_pre = list(chain.from_iterable(pre.data.tolist()))
            test_pred.extend(t_pre)
            test_y = list(chain.from_iterable(test_y.data.tolist()))
            test_label.extend(test_y)
        self.mae = mean_absolute_error(test_label, test_pred)
        self.rmse = np.sqrt(mean_squared_error(test_label, test_pred))

        test_pred = np.array(test_pred)
        test_label = np.array(test_label)
        test_result = dict({
            "test_label": test_label,
            "test_pre": test_pred
        })
        save_obj(test_result, os.path.join(self.save_path, name + '.pkl'))
        df_out_train = pd.DataFrame({"test_label": test_label, "test_pre": test_pred})
        df_out_train.to_csv(os.path.join(self.save_path, name + '.csv'), encoding="utf_8")

        return self.mae, self.rmse

# Run
for feature in features:
    result_path = os.path.join(result_path_prefix, feature)
    create_logger(result_path, "log.txt")
    os.makedirs(result_path, exist_ok=True)
    MAE, RMSE = [], []
    train_list = [os.path.join(battery_data_path, 'NCM{}.pkl'.format(id)) for id in train_id]
    val_list = [os.path.join(battery_data_path, 'NCM{}.pkl'.format(id)) for id in val_id]
    test_list = [os.path.join(battery_data_path, 'NCM{}.pkl'.format(id)) for id in test_id]
    if feature == 'VTPS':
        L = 5
    # trainloader
    train_dataset = MyDataset(train_list, feature)
    loader_train = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    log.info("Training samples: #{}".format(len(train_dataset)))
    # vailloader
    val_dataset = MyDataset(val_list, feature)
    loader_val = DataLoader(dataset=val_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
    log.info("Validating samples: #{}".format(len(val_dataset)))

    # train
    # [1070 10 400 5]
    net = CNN_LSTM(L=L, input_size=train_dataset.dimension, out_channels=out_channels, hidden_size=hidden_size, num_layers=num_layers).to(device)
    trainer = Trainer(feature=feature, epochs=epochs, lr=lr, weight_decay=weight_decay, save_path=result_path, patience=patience)
    net.train()
    log.info("Start training with feature={}".format(feature))
    trainer.train(net, loader_train, loader_val)

    # test
    net.load_state_dict(torch.load(os.path.join(result_path, "model.pt")))
    net.eval()
    for name in test_list:
        test_dataset = MyDataset([name], feature)
        Dtest = DataLoader(dataset=test_dataset, batch_size=4 * bs, shuffle=False, num_workers=num_workers)
        log.info("Testing samples: #{}".format(len(test_dataset)))
        name = os.path.split(name)[1]
        name = name.split('.')[0]
        mae, rmse = trainer.test(net, Dtest, name)
        MAE.append(mae)
        RMSE.append(rmse)
        log.info("{}: MAE={}, RMSE={}".format(name, mae, rmse))

    MAE = np.array(MAE)
    RMSE = np.array(RMSE)
    log.info("Test result @{}".format(feature))
    log.info("average MAE: {}, average RMSE: {}".format(np.mean(MAE), np.mean(RMSE)))
    destroy_logger()
