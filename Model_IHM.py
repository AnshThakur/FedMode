from unicodedata import bidirectional
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn.parameter import Parameter



## Prediction model

class LSTMClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, device, dropout_prob=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)  # LSTM Cell
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()  # For binary classification
        self.device = device
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)  # Dropout with a specified probability

    def forward(self, x):
        hidden, cell = self.init_hidden(x)
        time_steps = x.shape[1]  # shape of x is (batches, time_steps, features)
        
        for i in range(time_steps):
            inputs = x[:, i]  # (batch, features) shape
            hidden, cell = self.rnn(inputs, (hidden, cell))
            
        # Apply dropout before the final fully connected layer
        hidden = self.dropout(hidden)
        
        out = self.activation(self.fc1(hidden))  # Take the hidden vector corresponding to the last time step
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim)
        c0 = torch.zeros(x.size(0), self.hidden_dim)
        return h0.to(self.device), c0.to(self.device)








