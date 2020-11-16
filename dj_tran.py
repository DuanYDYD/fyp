import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from models import WaveNet
from utils import StockDataset, eval, scheduler, stock_data_paths, crypto_data_paths

def create_input_datadj(paths, n_lags, y_days=1):
    """ n_lags is days_length of input x
        y_days is days_length of output y
        use n_lags=25 and y_days=1 for mid-term first
     """

    X, y = [], []
    X_test, y_test = [[] for _ in range(len(paths))], [[] for _ in range(len(paths))]
    X_train, y_train = [[] for _ in range(len(paths))], [[] for _ in range(len(paths))]

    print("creating input data...")
    for i, path in enumerate(paths):
        # load data
        series_arr = pd.read_csv(path, index_col=0)
        # !!!! If the consecutive null values is > 2, interpolate function will do nothing
        # we drop those tickers with consecutive null values greater than 2
        series_arr = series_arr.interpolate().dropna(axis=1)
        series_arr = series_arr.values.T

        for series in series_arr:
            for step in range(len(series) - n_lags - y_days + 1):
                end_step = step + n_lags
                X.append(series[step:end_step])
                y.append(series[end_step:end_step+y_days])

        #0.8 for training, 0.2 for test
        valid_ind = int(len(X) * 0.8)

        #xxxxxxxxxxxxx 
        X_train_ = X[:valid_ind]
        y_train_ = y[:valid_ind]
  

        X_test_ = X[valid_ind:]
        y_test_ = y[valid_ind:]


        X_test[i].append(X_test_)
        y_test[i].append(y_test_)
        X_train[i].append(X_train_)
        y_train[i].append(y_train_)
        X =[]
        y =[]
    """ return a tensor with shape: X:(num_features, 1, num_samples, n_lags) 
                                    y:(num_features, 1, num_samples, y_days)"""
    return torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(X_test), torch.FloatTensor(y_test)

def train(net, N_EPOCHS, train_loader, LR, path):
    # initailize the network, optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir='runs/wave')

    running_loss = 0
    for epoch in range(N_EPOCHS):
        for i, (x_batch, y_batch) in enumerate(train_loader, 0):
            # prediction
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
        
            y_pred = net(x_batch)
            loss = criterion(y_pred, y_batch)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate = LR / (1 + ((i + 1) / 250))
            optimizer = scheduler(optimizer, learning_rate)

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                # ...log the running loss
                print("loss:", running_loss / 100, " batch:", (i + 1))
                writer.add_scalar('training loss wave{}'.format(datetime.today().strftime('%Y-%m-%d')),
                                     running_loss / 100, (i + 1) + epoch * 9747)
                running_loss = 0.0

    torch.save(net.state_dict(), path)
    writer.close()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(datetime.today().strftime('%Y-%m-%d'))
    print(f"Using {device}")
    # fix the random seed
    # 0 999 333 111 123
    SEED = 999

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # hyperparams
    BATCH_SIZE = 100
    N_EPOCHS = 3
    N_LAGS = 90
    Y_DAYS = 3
    NUM_WORKERS = 0
    LR = 0.01

    # model parameters
    layer_size = 3
    stack_size = 12
    in_channels = 6 # 6 features


    res_channels = 32

    # paths
    PATHS = stock_data_paths()
    MODEL_PATH = 'weights/djstock'

    net = WaveNet(layer_size, stack_size, in_channels, res_channels, Y_DAYS, N_LAGS)

    # load the dataset
    X_train, y_train, X_test, y_test = create_input_data(PATHS, N_LAGS, Y_DAYS)
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,     
                            batch_size=BATCH_SIZE)
    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,     
                            batch_size=BATCH_SIZE)

    train(net, N_EPOCHS, train_loader, LR, MODEL_PATH)
    eval(net, MODEL_PATH, test_loader)

    #The MSE is  12119.959969761985
