import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size_1), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size_1, hidden_size_2), nn.ReLU())
        self.fc3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out
