"""Custom layers."""
import os
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, init
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
from collections import OrderedDict
import torch.nn.functional as F

class ShareConv2d(nn.Module):
    def __init__(self, args, mask, imgsz, kernel_size, stride, p, bias=True):
        super(ShareConv2d, self).__init__()
        self.args = args.kernel_size
        self.mask = Parameter(mask, requires_grad=True)
        self.p_norm = Parameter(torch.zeros(p).add_(4), requires_grad=True)

        self.eps = 1e-60
        self.sigmoid = nn.Sigmoid()
        self.temperture = 5
        self.pooling_operation = "LSE"

        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=stride)
        self.fold = nn.Fold(output_size=((args.imgsz // args.kernel_size), (args.imgsz // args.kernel_size)), kernel_size=(1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(p))
        else:
            self.register_parameter("bias", None)
        self.ondo_w = "True"
        self.exp_p = "True"

    def pooling(self, x):
        """
        w:
            [shape]: (kernel size * 2, patch number)
        p:
            [shape]: (patch number)
        """
        if x[3] in ["train", "val"]:
            if self.ondo_w == "True":
                w_prime = self.sigmoid(self.mask.unsqueeze(0) * self.temperture)
            else:
                w_prime = self.sigmoid(self.mask.unsqueeze(0))
        elif x[3] == "test":
            if self.ondo_w == "True":
                step = STEP.apply
                w_prime = step(self.mask.unsqueeze(0) / self.temperture)
            else:
                step = STEP.apply
                w_prime = step(self.mask.unsqueeze(0))
        if self.exp_p == "True":
            p_prime = torch.exp(self.p_norm.unsqueeze(0).unsqueeze(0))
        else:
            p_prime = self.p_norm.unsqueeze(0).unsqueeze(0)

        if self.pooling_operation == "Non_LSE":
            z = x[1].to(torch.float64)
            z = self.unfold(z)
            z = z.reshape(z.shape[0], x[1].shape[1], self.kernel_size ** 2, z.shape[2])
            z = torch.abs(z)
            z = torch.pow(z + self.eps, p_prime)
            z = (z) * w_prime
            z = z.mean(dim=2, keepdim=True)
            z = torch.pow(z + self.eps, 1 / p_prime)
            z = z.reshape(z.shape[0], z.shape[1], z.shape[3])
            z = self.fold(z)
            z = z.to(torch.float32)
        elif self.pooling_operation =="LSE":
            p_prime = p_prime.unsqueeze(0)
            w_prime = w_prime.unsqueeze(0)
            z = x[1].to(torch.float64)
            z = self.unfold(z)
            z = z.reshape(x[1].shape[0],x[1].shape[1],self.args ** 2,z.shape[2])
            z = torch.abs(z)
            t = torch.log(w_prime + self.eps) + p_prime * torch.log(z + self.eps)
            t = torch.logsumexp(t + self.eps, dim=2, keepdim=True)
            t = (t - np.log(t.shape[2])) / p_prime
            t = torch.exp(t)
            z = t.reshape(t.shape[0], -1, t.shape[3])
            z = self.fold(z)
            z = z.to(torch.float32)
        return z, w_prime, x[2]

    def reset_parameters(self):
        init._no_grad_normal_(self.w, 0, 1)
        init._no_grad_normal_(self.p, 10)
        if self.bias is not None:
            y = self.pooling()
            fan_in, _ = init._calculate_fan_in_and_fan_out(y)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        z, w, weight1 = self.pooling(x)
        return z, w, weight1

class MetaLinearModel(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.mask = torch.rand(args.kernel_size ** 2, args.p)
        self.first_layer = nn.Sequential(layers.ShareConv2d(args, self.mask, args.imgsz, args.kernel_size, args.stride, args.p, bias=False)).to(device)
        self.fc4_layers = nn.Sequential(
                                nn.Linear(1 * (args.imgsz // args.kernel_size) * (args.imgsz // args.kernel_size), 128, bias=False),
                                nn.BatchNorm1d(num_features=128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 64, bias=False),
                                nn.BatchNorm1d(num_features=64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 5, bias=False),
                                nn.BatchNorm1d(num_features=5),
                                nn.ReLU(inplace=True),
                            ).to(device)
        def forward(self, x):
            x, w, weight = self.first_layer(x)
            x = x.view(x.size(0), -1)
            x1 = self.fc4_layers(x)
            return x1, w

def conv_block(in_channels, hidden_size):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1, bias=False)),
        ('norm', nn.BatchNorm2d(hidden_size, momentum=1.0, track_running_stats=False)),
        ('relu', nn.ReLU()),
    ]))
def maxpool():
    return nn.MaxPool2d(2, stride=2)

class MetaConvModel(nn.Module):
    def __init__(self, args):
        super(MetaConvModel, self).__init__()
        self.mask = torch.rand(args.kernel_size ** 2, args.p)
        self.first_layer = nn.Sequential(ShareConv2d(args, self.mask, args.imgsz, args.kernel_size, args.stride, args.p, bias=False))
        self.second_layer = nn.Sequential(
                                nn.Linear(args.hidden_size * (args.imgsz // args.kernel_size) * (args.imgsz // args.kernel_size), 512, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 5, bias=False),
                                nn.ReLU(inplace=True),
                            )
        self.features = nn.Sequential(OrderedDict([
            ('layer1', conv_block(in_channels=1, hidden_size=args.hidden_size)),]))

    def forward(self, x):
        out = self.features(x[1])
        x[1] = out
        x, w, weight1 = self.first_layer(x)
        x = x.view(x.size(0), -1)
        x1 = self.second_layer(x)
        return x1, w