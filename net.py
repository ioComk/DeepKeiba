import torch as t
from torch import nn

INPUT_UNITS  = 270
HIDDEN_UNITS1 = 128
HIDDEN_UNITS2 = 32
OUT_UNITS = 18

class Model(nn.Module):

    def __init__(self, dropout=0):
        super(Model, self).__init__()
        
        p = dropout

        # Dense
        self.dense = nn.Sequential(
            nn.Linear(INPUT_UNITS, HIDDEN_UNITS1),
            nn.Dropout(p),
            nn.BatchNorm1d(HIDDEN_UNITS1),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS1, HIDDEN_UNITS1),
            nn.Dropout(p),
            nn.BatchNorm1d(HIDDEN_UNITS1),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS1, HIDDEN_UNITS2),
            nn.Dropout(p),
            nn.BatchNorm1d(HIDDEN_UNITS2),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS2, HIDDEN_UNITS2),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS2, OUT_UNITS),
        )

    def forward(self, x):
        
        y = self.dense(x)

        return y