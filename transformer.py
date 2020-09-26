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

from tst import Transformer
from utils import create_input_data, StockDataset, eval

def train(N_EPOCHS, train_loader):
    # initailize the network, optimizer and loss function
    net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
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

            if i == 1:
                print(x_batch.size())
                print(y_batch.size())
                print(y_pred.size())

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                # ...log the running loss
                print("loss:", running_loss / 1000, " seq:", i + 1)
                writer.add_scalar('training loss with)', running_loss / 1000, i + 1)
                running_loss = 0.0

    torch.save(net.state_dict(), "weights/trans")
    writer.close()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    BATCH_SIZE = 100
    N_EPOCHS = 1
    N_LAGS = 25
    NUM_WORKERS = 2
    LR = 0.001

    # Model parameters
    d_model = 32  # Lattent dim
    q = 8  # Query size
    v = 8  # Value size
    h = 8  # Number of heads
    N = 4  # Number of encoder and decoder to stack
    attention_size = 12  # Attention window size
    dropout = 0.2  # Dropout rate
    pe = None  # Positional encoding
    chunk_mode = None

    # features sizes
    d_input = 6  # adj_close, open, close, high, low, volume
    d_output = 1 # only predict adj_close

    # load the dataset
    X_train, y_train, X_test, y_test = create_input_data(PATHS, N_LAGS, transformer=True)
    train_dataset = StockDataset(X_train, y_train, transformer=True)
    train_loader = DataLoader(dataset=train_dataset,     
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS)
    test_dataset = StockDataset(X_test, y_test, transformer=True)
    test_loader = DataLoader(dataset=test_dataset,     
                            batch_size=BATCH_SIZE)

    #train(N_EPOCHS, train_loader)

    net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
    eval(net, "weights/trans" ,test_loader)
    