"""
This script provides an entry point to step through the symbolic 
tracing functionality of `torch.fx`.
"""

# Notes:
# * In VSCode, remember to add `"justMyCode": false` to "launch.json"

import torch
from torch import nn, tensor
import torch.nn.functional as F
import torch.fx as fx
import inspect

class SimpleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = SimpleCnn()
broken_model = SimpleCnn()
broken_model.fc1 = nn.Linear(9216, 120)

traced_model = fx.symbolic_trace(model)
traced_model.graph.print_tabular()

# traced_broken_model = fx.symbolic_trace(model)
# traced_broken_model.graph.print_tabular()

