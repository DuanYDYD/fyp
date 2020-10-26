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

from models import Transformer
from utils import create_input_data, StockDataset, eval, scheduler, plot_one_stock

def train(net, N_EPOCHS, train_loader, LR, path):
    # initailize the network, optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir='runs/trans')

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
                writer.add_scalar('training loss trans{}'.format(datetime.today().strftime('%Y-%m-%d')),
                                     running_loss / 100, (i + 1) + epoch * len(train_loader))
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
    MODEL_PATH = 'weights/trans/trans_12632se5'
    # hyperparams
    BATCH_SIZE = 100
    N_EPOCHS = 3
    N_LAGS = 25
    NUM_WORKERS = 0
    LR = 0.01

    # model parameters
    dim_input = 6
    output_sequence_length = 1
    dec_seq_len = 2
    dim_val = 32
    dim_attn = 12
    n_heads = 8 
    n_encoder_layers = 12
    n_decoder_layers = 6
    #init network
    net = Transformer(dim_val, dim_attn, dim_input, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)

    # load the dataset
    X_train, y_train, X_test, y_test = create_input_data(PATHS, N_LAGS)
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,     
                            batch_size=BATCH_SIZE)
    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,     
                            batch_size=BATCH_SIZE)

    #train(net, N_EPOCHS, train_loader, LR, MODEL_PATH)
    eval(net, MODEL_PATH, test_loader)
    #tensorboard --logdir=runs
    plot_one_stock(X_test, y_test, net, MODEL_PATH)
    #The MSE is  0.0003273937825206491 ff
    #The MSE is  0.0001812670836877635 33 10
    #The MSE is  0.0003356801978449813 64   10
    #The MSE is  0.0001615765498885661964 64 32
    #The MSE is  0.0005471179923741005 64 64
    #The MSE is  0.00015378855410287437 33 32
    #The MSE is  0.00025121048723348866 126 32
    #the MSE is  0.00020284409984236388 21 32

    #The MSE is  9.102330600960862e-05 42 32 se1
    #The MSE is  0.0003441643159479967 42 32 se2
    #the MSE is  4.66587838643595e-05 42 32 se3
    #The MSE is  0.00013140986475010456 se4
    #The MSE is  9.675200523454657e-05