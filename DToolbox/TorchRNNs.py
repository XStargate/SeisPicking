import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F

class TRNN(nn.Module):
    def __init__(self):
        super(TRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=20, 
                            num_layers=1, batch_first=True)
        self.fc  = nn.Linear(20*50, 2)

    def forward(self, x):
        x, hn = self.rnn(x, None)
        x = x.reshape(x.size()[0], x.size()[1]*x.size()[2])
        x = self.fc(x)
        return x