import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(35, 70), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(70, 35), nn.ReLU())
        self.fc3 = nn.Linear(35, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out
