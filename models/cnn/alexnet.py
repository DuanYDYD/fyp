import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, n_lags, y_days):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=15, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, y_days),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2048)
        x = self.classifier(x)
        return x

    #The MSE is  0.0006266263563656976