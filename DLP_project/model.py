# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:10:22 2022

@author: user
"""

import torch
import torch.nn as nn
import numpy

class six_layer_CNN(nn.Module):
    def __init__(self):
        super(six_layer_CNN, self).__init__()
        self.main = nn.Sequential(
            # 64 * 1 * 750
            nn.Conv1d(1, 64, 21, 5),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # 64 * 64 * 146
            nn.Conv1d(64, 64, 21, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # 64 * 64 * 126
            nn.MaxPool1d(2, 2),
            # 64 * 64 * 63
            
            nn.Conv1d(64, 128, 5, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 5, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            # 64 * 128 * 27
            
            nn.Conv1d(128, 256, 5, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 5, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            # 64 * 256 * 19
            
            nn.AvgPool1d(19)
            # 64 * 256 * 1
        )
        
        self.fc = nn.Linear(256, 4)
        

    def forward(self, x):
        output = self.main(x)
        output = output.view(-1, 256)
        output = self.fc(output)
        return output
        





