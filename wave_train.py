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
from utils import create_input_data, StockDataset, eval, scheduler, plot_one_stock

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
    torch.manual_seed(123)
    np.random.seed(123)

    # paths
    PATHS = ['data/sp500_joined_adj.csv', 
            'data/sp500_joined_open.csv',
            'data/sp500_joined_close.csv',
            'data/sp500_joined_high.csv',
            'data/sp500_joined_low.csv',
            'data/sp500_joined_volume.csv']
    MODEL_PATH = 'weights/wave/wave_2832se5'
    # hyperparams
    BATCH_SIZE = 100
    N_EPOCHS = 3
    N_LAGS = 25
    NUM_WORKERS = 0
    LR = 0.01

    # model parameters
    layer_size = 2
    stack_size = 8
    in_channels = 6 # 6 features
    res_channels = 32

    net = WaveNet(layer_size, stack_size, in_channels, res_channels)

    # load the dataset
    X_train, y_train, X_test, y_test = create_input_data(PATHS, N_LAGS)
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,     
                            batch_size=BATCH_SIZE)
    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,     
                            batch_size=BATCH_SIZE)

    train(net, N_EPOCHS, train_loader, LR, MODEL_PATH)
    eval(net, MODEL_PATH, test_loader)
    plot_one_stock(X_test, y_test, net, MODEL_PATH)
    #The MSE is  0.0002094345309450686 28 64
    #The MSE is  0.0004989148894638609 27 32
    #The MSE is  0.00017359222582504684 25 32
    #The MSE is  0.00017477968110319773 22 32
    #The MSE is  0.00011558366908265208 41 32

    #The MSE is  8.248745888667056e-05 28 32 se1
    #The MSE is  9.72329971868175e-05 28 32 se2
    #The MSE is  5.453151254948243e-05 28 32 se3
    #The MSE is  0.00016852398847098173 28 32 se4
    #The MSE is  0.00017561992712846868 28 32 se5

    #The MSE is  9.702412049717072e-05 33 32
    #The MSE is  0.0006524808009704072 32 32
    #The MSE is  0.00016231196748245488 33 32 se2
    #The MSE is  6.993623421993204e-05 33 32 se3
    #The MSE is  0.00046851240992866765 33 32 se4
    #The MSE is  0.0002352291541823431 33 32 se5
    

