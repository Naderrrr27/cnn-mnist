import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "cnn_mnist.pth"

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_feats=False):
        x = F.relu(self.conv1(x))
        feat1 = x
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        feat2 = x        
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if return_feats:
            return x, feat1.detach().cpu(), feat2.detach().cpu()
        return x
