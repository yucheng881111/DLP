import torch
import torch.nn as nn
import numpy

class DeepConvNet(nn.Module):
    def __init__(self, act_func, device):
        super(DeepConvNet, self).__init__()
        self.device = device
        self.conv0 = nn.Conv2d(1, 25, kernel_size = (1, 5))
        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size = (2, 1)),
            nn.BatchNorm2d(25, eps = 1e-5, momentum = 0.1),
            act_func,
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout(p = 0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size = (1, 5)),
            nn.BatchNorm2d(50, eps = 1e-5, momentum = 0.1),
            act_func,
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout(p = 0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size = (1, 5)),
            nn.BatchNorm2d(100, eps = 1e-5, momentum = 0.1),
            act_func,
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout(p = 0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size = (1, 5)),
            nn.BatchNorm2d(200, eps = 1e-5, momentum = 0.1),
            act_func,
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout(p = 0.5)
        )

        self.classify = nn.Linear(8600, 2)

    def forward(self, X):
        out = self.conv0(X.to(self.device))
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.shape[0], -1)  #flatten
        out = self.classify(out)
        #out = nn.functional.softmax(out, dim = 0)
        return out

