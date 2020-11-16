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
from utils import create_input_data, StockDataset, eval, scheduler, stock_data_paths

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
    dim_input = 6
    output_sequence_length = Y_DAYS
    dec_seq_len = Y_DAYS
    dim_val = 32
    dim_attn = 12
    n_heads = 8 
    n_encoder_layers = 4
    n_decoder_layers = 2

    # paths
    PATHS = stock_data_paths()
    MODEL_PATH = 'weights/trans/stock/3days/{e}_{d}_{v}_{y}_seed{seed}'.format(e=n_encoder_layers, d=n_decoder_layers, v=dim_val, y=Y_DAYS, seed=SEED)

    #init network
    net = Transformer(dim_val, dim_attn, dim_input, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)

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
    #tensorboard --logdir=runs

    #The MSE is  0.0010176260894923762 0
    #The MSE is  0.0008223066139244062 333
    #The MSE is  0.0010279362977172954 111 ep 3 ss
    #The MSE is  0.0008899663603104443 123 ep 3 ss
    #The MSE is  0.0008842754231835146 999 dim1 5 epoch

    #Multi horizon 3 days
    #The MSE is  0.006249775759577008 se0
    #The MSE is  0.002294417190863021 s999
    

   
