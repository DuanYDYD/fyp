import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from tst import Transformer
from utils import create_input_data, StockDataset, eval

def train(net, N_EPOCHS, train_loader, LR):
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

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                # ...log the running loss
                print("loss:", running_loss / 100, " batch:", (i + 1))
                writer.add_scalar('training loss trans{}'.format(datetime.today().strftime('%Y-%m-%d')),
                                     running_loss / 100, (i + 1) + epoch * 9700)
                running_loss = 0.0

    torch.save(net.state_dict(), "weights/trans")
    writer.close()

def plot_one_stock(X_test, y_test, net, path, transformer=False):
    stock_idx = 0
    X_test = X_test[:, stock_idx, :, :] # choose the first stock
    X_test = X_test.transpose(0, 1).transpose(1, 2)
    y_test = y_test[0][stock_idx]
    net.load_state_dict(torch.load(path))
    net.eval()

    with torch.no_grad():
        if transformer:
            y_test = y_test[:, -1]
            y_pred = net(X_test)
            print(y_pred.size())
            y_pred = y_pred[:, -1, :].squeeze(0)
            print(y_pred.size())
        else:
            y_pred = net(X_test)

    time_step = range(len(y_test))
    plt.plot(time_step, y_pred, label='Prediction')
    plt.plot(time_step, y_test, label='Actual price')
    plt.title("one stock grpah")
    plt.ylabel("normalized adjusted close price")
    plt.xlabel("time step")
    plt.legend()
    #plt.savefig("stock.png")
    plt.show() 

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(datetime.today().strftime('%Y-%m-%d'))
    print(f"Using {device}")
    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # train parameters
    PATHS = ['data/sp500_joined_adj.csv', 
            'data/sp500_joined_open.csv',
            'data/sp500_joined_close.csv',
            'data/sp500_joined_high.csv',
            'data/sp500_joined_low.csv',
            'data/sp500_joined_volume.csv']
    MODEL_PATH = 'weights/trans'
    # hyperparams
    BATCH_SIZE = 100
    N_EPOCHS = 3
    N_LAGS = 25
    NUM_WORKERS = 0
    LR = 0.002

    # model parameters
    dim_input = 6
    output_sequence_length = 1
    dec_seq_len = 2
    dim_val = 10
    dim_attn = 12
    n_heads = 8 
    n_decoder_layers = 4
    n_encoder_layers = 6
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

    train(net, N_EPOCHS, train_loader, LR)
    eval(net, MODEL_PATH, test_loader)
    plot_one_stock(X_test, y_test, net, MODEL_PATH)