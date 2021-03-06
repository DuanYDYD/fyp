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
from utils import create_input_data, StockDataset, eval, scheduler, stock_data_paths, crypto_data_paths

def train(net, N_EPOCHS, train_loader, LR, path):
    # initailize the network, optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    #writer = SummaryWriter(log_dir='runs/wave')

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
            #learning_rate = LR / (1 + ((i + 1) / 250))
            #optimizer = scheduler(optimizer, learning_rate)

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                # ...log the running loss
                print("loss:", running_loss / 50, " batch:", (i + 1))
                running_loss = 0.0
        print("epoch ", epoch)
    torch.save(net.state_dict(), path)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(datetime.today().strftime('%Y-%m-%d'))
    print(f"Using {device}")
    # fix the random seed
    # 0 999 333 111 123
    SEED = 0

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # hyperparams
    BATCH_SIZE = 100
    N_EPOCHS = 400#20 400
    N_LAGS = 90
    Y_DAYS = 3
    NUM_WORKERS = 0
    LR = 0.0001 #5

    # model parameters
    layer_size = 2
    stack_size = 29
    in_channels = 6 # 6 features
    res_channels = 64

    # paths
    PATHS = crypto_data_paths()
    MODEL_PATH = 'weights/wave/crypto/3days/{l}_{s}_{r}_{y}_{seed}'.format(l=layer_size, s=stack_size, r=res_channels, y=Y_DAYS, seed=SEED)

    net = WaveNet(layer_size, stack_size, in_channels, res_channels, Y_DAYS, N_LAGS)
    net.load_state_dict(torch.load(MODEL_PATH))

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


    #The MSE is  0.00040897312399241256 28 32 se123 1
    #The MSE is  0.0007783590546027762 28 32 se111 1
    #The MSE is  0.0003596830506049608 se 333
    #The MSE is  0.00013038388036484518 se 999
    #The MSE is  0.0007731251604861855   


    #The MSE is  0.00030988159688583 25 se0
    #The MSE is  0.00024071033664234363 25 se999
    #The MSE is  0.0003270250107385795 25 se 333
    #The MSE is  0.013896949215793887 25 se 111
    #The MSE is  0.013603525761472259 27
    #The MSE is  0.0010279362977172954

    #3days
    #The MSE is  0.0021889363289384965 3 12

    #crypto 
    #one The MSE is  3.615179643878579e-05 64
    #one The MSE is  0.00010890758233900188 32 
    #3days The MSE is  0.0015587311456627848 64 3 12
    #The MSE is  0.0016606460898516138