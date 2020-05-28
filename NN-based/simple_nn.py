import sys
sys.path.append("..")
import Preprocess.make_dataset as md
import Preprocess.csv_utils as cu
import Preprocess.meta_data_2018 as md8

import os 
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as Data 
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import argparse

def get_data(file_path):
    raw_data = cu.load_all_file_data(file_path, True)
    raw_data = cu.drop_attribute(raw_data, ['Timestamp'])
    raw_data = raw_data.dropna(axis=0)
    raw_data = md.convert_datatype(raw_data, "ids2018")
    raw_data = shuffle(raw_data)
    raw_data = raw_data.replace([np.inf, -np.inf], np.nan)
    raw_data = raw_data.dropna(axis=0)
    print(raw_data.shape)
    print(raw_data.isnull().values.any())
    LE = LabelEncoder()
    values = raw_data.values
    X, Y = values[:, :-1], values[:, -1]
    all_classes = np.unique(Y)
    LE.fit(all_classes)
    Y = LE.transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print("Size of X_train: ", X_train.shape)
    print("Size of X_test: ", X_test.shape)
    print("Size of Y_train: ", Y_train.shape)
    print("Size of Y_test: ", Y_test.shape)

    return LE, X_train, Y_train, X_test, Y_test

class AutoEncoder(nn.Module):
    def __init__(self, num_features):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(num_features, 40),
                                     nn.ReLU(True),
                                     nn.Linear(40, 20),
                                     nn.ReLU(True),
                                     nn.Linear(20, 10))
        self.decoder = nn.Sequential(nn.Linear(10, 20),
                                     nn.ReLU(True),
                                     nn.Linear(20, 40),
                                     nn.ReLU(True),
                                     nn.Linear(40, num_features),
                                     nn.Tanh())
    
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(x)
        return encode, decode

if __name__ == "__main__":
    file_path = "../../split_csv_new/"
    batch_size = 128
    lr = 0.01
    wd = 1e-5
    epoches = 40

    LE, X_train, Y_train, X_test, Y_test = get_data(file_path)
    num_features = X_train.shape[-1]
    train_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    test_loader = Data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    iter_train = iter(train_loader)

    AE = AutoEncoder(num_features)
    criterion = nn.MSELoss()
    optimizier = torch.optim.Adam(AE.parameters(), lr=lr, weight_decay=wd)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    AE = AE.to(device)

    for epoch in range(epoches):
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1
        
        for train_batch, _ in train_loader:
            _, output = AE(train_batch)
            loss = criterion(output, train_batch)

            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

        print("epoch: {}, loss is {}".format((epoch+1), loss.data.float()))
