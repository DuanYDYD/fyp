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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_input_data(paths, n_lags, transformer=False):
    # dividve the series(N=7)
    X, y = [], []
    X_test, y_test = [[] for _ in range(len(paths))], [[] for _ in range(len(paths))]
    X_train, y_train = [[] for _ in range(len(paths))], [[] for _ in range(len(paths))]

    for i, path in enumerate(paths):
        # load data
        series_arr = pd.read_csv(path, index_col=0)
        # !!!! If the consecutive null values is > 2, interpolate function will do nothing
        # we drop those tickers with consecutive null values greater than 2
        series_arr = series_arr.interpolate().dropna(axis=1)
        series_arr = series_arr.values.T

        # normalize the time series
        max_ = np.amax(series_arr)
        min_ = np.amin(series_arr)
        series_arr = (series_arr - min_) / (max_ - min_)

        #0.8 for training, 0.2 for test
        valid_ind = int(len(series_arr[0]) * 0.8)

        for series in series_arr:
            for step in range(len(series) - n_lags):
                end_step = step + n_lags
                X.append(series[step:end_step])
                if transformer:    
                    y.append(series[step+1 :end_step+1])
                else:
                    y.append(series[end_step])

            X_test[i].append(X[valid_ind:])
            y_test[i].append(y[valid_ind:])
            X_train[i].append(X[:valid_ind])
            y_train[i].append(y[:valid_ind])
            X =[]
            y =[]
    """ return a tensor with shape: X:(num_features, num_tickers, each_ticker_len, n_lags) 
                                    y:(num_features, num_tickers, each_ticker_len, 1)
        if transformer =True:
                                 y:(num_features, num_tickers, each_ticker_len, n_lags) """
    return torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(X_test), torch.FloatTensor(y_test)

class StockDataset(Dataset):
    """ return (X, y) 
        X : (n_lags, features(6))
        y: (n_lags, adj_close)
     """
    def __init__(self, X, y, transformer=False):
        super().__init__()
        if transformer:
            self.X = torch.flatten(X, start_dim=1, end_dim=2).transpose(0, 1).transpose(1, 2)
            self.y = torch.flatten(y[0], end_dim=1).unsqueeze(dim=2)
        else:
            self.X = torch.flatten(X, start_dim=1, end_dim=2).transpose(0, 1).transpose(1, 2)
            self.y = torch.flatten(y[0], end_dim=1).unsqueeze(dim=1)
        print("Input data Size: ", self.X.size())
        print("Label Size: ", self.y.size())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return self.X.shape[0]

def eval(net, path ,test_loader):
    # evalute the model and plot the graph of a choosing stock
    net.load_state_dict(torch.load(path))
    criterion = nn.MSELoss(reduction = 'sum') #Square error
    net.eval()

    total_loss = 0.0
    prices_pred = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            y_pred = net(x)
            total_loss += criterion(y_pred, y).item()
            if i % 1000 == 999:
                print("processing")
        mse = total_loss / len(test_loader.dataset)
    print("The MSE is ", mse)

class Net(nn.Module):
    # simple forward network
    def __init__(self, n_lags):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20 * n_lags, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(torch.flatten(x, start_dim=1))
        return x

def train(net, N_EPOCHS, train_loader):
    # initailize the network, optimizer and loss function
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
            loss = criterion(y_pred, y_batch)

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


if __name__ == "__main__":
    # sample framework
    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # hyperparameters
    PATHS = ['data/sp500_joined_adj.csv', 
            'data/sp500_joined_open.csv',
            'data/sp500_joined_close.csv',
            'data/sp500_joined_high.csv',
            'data/sp500_joined_low.csv',
            'data/sp500_joined_volume.csv']
    BATCH_SIZE = 10
    N_EPOCHS = 1
    N_LAGS = 25

    # load the dataset
    X_train, y_train, X_test, y_test = create_input_data(PATHS, N_LAGS)
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,     
                            batch_size=BATCH_SIZE)
    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,     
                            batch_size=BATCH_SIZE)
    net = Net(N_LAGS).to(device)
    #train(net, N_EPOCHS, train_loader)
    eval(net, path="weights/ff" , test_loader=test_loader)

    
