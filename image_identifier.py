from torch import nn, flatten
from torch.nn import functional as F

# https://www.cs.toronto.edu/~lczhang/360/lec/w05/overfit.html


class Image_Identifier(nn.Module):
    def __init__(self):
        super(Image_Identifier, self).__init__()

        dropout = .4

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=1), # 32x32
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(2, 2), # 16x16
            nn.Conv2d(8, 16, (3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(2, 2), # 8x8
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(2, 2), # 4x4
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(2, 2), # 2x2
        )

        self.fc = nn.Sequential(
            nn.Linear(2*2*64, 32),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Linear(32, 1),
        )
    
    def forward(self, X):
        X = self.conv(X)

        X = flatten(X, 1)
        X = self.fc(X)
        return X