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
from models import AlexNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_input_data(paths, n_lags, y_days=1):
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

        # normalize the time series
        X_train_ = X[:valid_ind]
        y_train_ = y[:valid_ind]
  
        
        max_ = max(np.amax(X_train_), np.amax(y_train_))
        min_ = min(np.amin(X_train_), np.amin(y_train_))
        X_train_ = (X_train_ - min_) / (max_ - min_)
        y_train_ = (y_train_ - min_) / (max_ - min_)

        X_test_ = X[valid_ind:]
        y_test_ = y[valid_ind:]
        max_ = max(np.amax(X_test_), np.amax(y_test_))
        min_ = min(np.amin(X_test_), np.amin(y_test_))
        X_test_ = (X_test_ - min_) / (max_ - min_)
        y_test_ = (y_test_ - min_) / (max_ - min_)

        X_test[i].append(X_test_)
        y_test[i].append(y_test_)
        X_train[i].append(X_train_)
        y_train[i].append(y_train_)
        X =[]
        y =[]
    """ return a tensor with shape: X:(num_features, 1, num_samples, n_lags) 
                                    y:(num_features, 1, num_samples, y_days)"""
    return torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(X_test), torch.FloatTensor(y_test)

class StockDataset(Dataset):
    """ return (X, y) 
        X : (n_lags, features(6))
        y: (n_lags, adj_close)
     """
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.flatten(X, start_dim=1, end_dim=2).transpose(0, 1).transpose(1, 2)
        self.y = torch.flatten(y[0], end_dim=1)
        print("Input data Size: ", self.X.size())
        print("Label Size: ", self.y.size())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return self.X.shape[0]

def eval(net, path, test_loader):
    # evalute the model using MSE
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

def plot_one_stock(X_test, y_test, net, path, length):
    # wrong function need to be fix!
    X_test = X_test[:, 0, :, :] # choose the first stock
    X_test = X_test.transpose(0, 1).transpose(1, 2)[:length]
    y_test = y_test[0, 0, :length, :]
 
    net.load_state_dict(torch.load(path))
    net.eval()

    with torch.no_grad():
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

def scheduler(optimizer,lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class Net(nn.Module):
    # simple forward network
    def __init__(self, n_lags, num_fea, y_days):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_fea * n_lags, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, y_days)

    def forward(self, x):
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x

def train(net, N_EPOCHS, train_loader):
    # initailize the network, optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
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
            if (i + 1) % 100 == 0:
                # ...log the running loss
                print("loss:", running_loss / 100, " batch:", (i + 1))
                writer.add_scalar('training loss ff{}'.format(datetime.today().strftime('%Y-%m-%d')),
                                     running_loss / 100, (i + 1) + epoch * 9747)
                running_loss = 0.0

    torch.save(net.state_dict(), "weights/ff")
    writer.close()

def stock_data_paths():
    paths = ['data/stock_data/sp500_joined_adj.csv', 
            'data/stock_data/sp500_joined_open.csv',
            'data/stock_data/sp500_joined_close.csv',
            'data/stock_data/sp500_joined_high.csv',
            'data/stock_data/sp500_joined_low.csv',
            'data/stock_data/sp500_joined_volume.csv']
    return paths

def crypto_data_paths():
    paths = [
            'data/crypto_data/coins_joined_open.csv',
            'data/crypto_data/coins_joined_close.csv',
            'data/crypto_data/coins_joined_high.csv',
            'data/crypto_data/coins_joined_low.csv',
            'data/crypto_data/coins_joined_volumeto.csv',
            'data/crypto_data/coins_joined_volumefrom.csv', ]
    return paths

def oil_data_paths():
    paths = [
            'data/oil_data/oil_price.csv',]
    return paths

if __name__ == "__main__":
    # sample framework
    # fix the random seed
    print(datetime.today().strftime('%Y-%m-%d'))
    # 0 999 333 111 123
    torch.manual_seed(0)
    np.random.seed(0)

    # hyperparameters
    PATHS = stock_data_paths()
    MODEL_PATH="weights/ff"
    BATCH_SIZE = 100
    N_EPOCHS = 1
    N_LAGS = 25
    Y_DAYS = 1
    NUM_FEA = 1

    # load the dataset
    X_train, y_train, X_test, y_test = create_input_data(PATHS, N_LAGS, Y_DAYS)
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,     
                            batch_size=BATCH_SIZE)
    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,     
                            batch_size=BATCH_SIZE)

    net = AlexNet(N_LAGS, Y_DAYS).to(device)
    train(net, N_EPOCHS, train_loader)
    eval(net, MODEL_PATH, test_loader=test_loader)
    #plot_one_stock(X_test, y_test, net, MODEL_PATH, length=1000)

    #The MSE is  0.002010555131915416
    #The MSE is  0.004173629942736138
    

