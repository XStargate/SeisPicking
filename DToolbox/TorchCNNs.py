import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F

class TCNN(nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 2)
        self.conv2 = nn.Conv2d(1, 1, 2)
        self.pool  = nn.MaxPool2d((2, 2), ceil_mode=True)
        self.fc   = nn.Linear(600, 2)

    def create_fc_layer(self, num_inputs, num_outputs=2):
        self.fc = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        num_features = self.num_flat_features(x)
        x = x.view(-1, num_features)
        x = self.fc(x)

        return x

    def num_flat_features(self, layer):
        return layer.size()[2] * layer.size()[3]