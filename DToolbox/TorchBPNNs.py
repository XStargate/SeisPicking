import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F

class BPNN(nn.Module):
    def __init__(self):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

