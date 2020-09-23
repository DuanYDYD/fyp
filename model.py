import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import (Dataset, TensorDataset, 
                             DataLoader, Subset)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    # simple forward network
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



def train(N_EPOCHS, train_loader):
    # initailize the network, optimizer and loss function
    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir='runs/ff')

    running_loss = 0
    for epoch in range(N_EPOCHS):
        for i, (x_batch, y_batch) in enumerate(train_loader, 0):
            # prediction
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = net(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)

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

    torch.save(net.state_dict(), "weights/ff")
    writer.close()

def create_input_data(series_arr, n_lags=7):
    # dividve the series(N=7)
    X, y = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []

    #0.8 for training, 0.2 for test
    valid_ind = int(len(series_arr[0]) * 0.8)

    for series in series_arr:
        for step in range(len(series) - n_lags):
            end_step = step + n_lags
            X.append(series[step:end_step])
            y.append(series[end_step])

        X_test.append(X[valid_ind:])
        y_test.append(y[valid_ind:])
        X_train.append(X[:valid_ind])
        y_train.append(y[:valid_ind])
        X =[]
        y =[]
    
    # return a tensor with shape: X:(num_tickers, each_ticker_len, 7) y:(num_tickers, each_ticker_len, 1)
    return torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(X_test), torch.FloatTensor(y_test)

def mean_square_error(prices, X_test, y_test):
    # evalute the model and plot the graph of a choosing stock
    net = Net()
    net.load_state_dict(torch.load("weights/ff"))
    
    prices_pred = []
    with torch.no_grad():
        for x in torch.flatten(X_test, end_dim=1):
            y_pred = net(x)
            prices_pred.append(y_pred.numpy())

    pred = np.squeeze(np.array(prices_pred))
    actual = torch.flatten(y_test, end_dim=1).numpy()
    print(np.sum((pred - actual) ** 2) / len(pred))


def plot_one_stock(prices, X_test, y_test, stock_num):
    # plot the graph of a choosing stock
    net = Net()
    net.load_state_dict(torch.load("weights/ff"))

    X_test = X_test[stock_num]
    y_test = y_test[stock_num]
    one_stock_pred = []

    with torch.no_grad():
        for x in X_test:
            y_pred = net(x)
            one_stock_pred.append(y_pred)
       
    time_step = range(len(y_test))
    plt.plot(time_step, y_test, label='True')
    plt.plot(time_step, one_stock_pred, label='Prediction')
    plt.title("Movement")
    plt.ylabel("Stock Price")
    plt.xlabel("Time step")
    plt.legend()
    plt.savefig("neural.png")
    plt.show() 



if __name__ == "__main__":
    # load data
    prices = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    # !!!! If the consecutive null values is > 2, interpolate function will do nothing
    # we drop those tickers with consecutive null values greater than 2
    prices = prices.interpolate().dropna(axis=1)
    prices = prices.values.T

    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # hyperparameters
    BATCH_SIZE = 10
    N_EPOCHS = 3
    N_LAGS = 7

    # load the dataset
    X_train, y_train, X_test, y_test = create_input_data(prices)
    train_dataset = TensorDataset(torch.flatten(X_train, end_dim=1), torch.flatten(y_train, end_dim=1))
    train_loader = DataLoader(dataset=train_dataset,     
                            batch_size=BATCH_SIZE)

    #train(N_EPOCHS, train_loader)
    #mean_square_error(prices, X_test, y_test)
    plot_one_stock(prices, X_test, y_test, 2)