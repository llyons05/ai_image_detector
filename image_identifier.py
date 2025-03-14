from torch import nn, flatten

# https://www.cs.toronto.edu/~lczhang/360/lec/w05/overfit.html
# https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images dataset with larger images

class Image_Identifier(nn.Module):
    def __init__(self):
        super(Image_Identifier, self).__init__()

        dropout = .45

        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, (3, 3), padding=1, padding_mode="replicate"), # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(2, 2), # 16x16
            nn.Conv2d(128, 64, (3, 3), padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(2, 2), # 8x8
            nn.Conv2d(64, 32, (3, 3), padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(2, 2), # 4x4
            nn.Conv2d(32, 16, (3, 3), padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(2, 2), # 2x2
        )

        self.fc = nn.Sequential(
            nn.Linear(2*2*16, 32),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Linear(32, 1),
        )
    
    def forward(self, X):
        X = self.conv(X)

        X = flatten(X, 1)
        X = self.fc(X)
        return X