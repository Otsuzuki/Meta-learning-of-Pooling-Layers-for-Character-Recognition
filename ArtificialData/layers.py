"""Custom layers."""
import os
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, init
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
import torch.nn.functional as F

class lp_pooling1d(nn.Module):
    def __init__(self, mask, args, bias=True):
        super(lp_pooling1d, self).__init__()
        self.mask = Parameter(mask, requires_grad=True)
        self.p_norm = Parameter(torch.zeros(args.output_size).add_(2), requires_grad=True)
        self.eps = 1e-16
        self.temperture = 5
        self.sigmoid = nn.Sigmoid()
        self.pooling_operation = "Non_LSE"
        if bias:
            self.bias = Parameter(torch.Tensor(30))
        else:
            self.register_parameter("bias", None)

    def pooling(self, x):
        if self.pooling_operation == "Non_LSE":
            w_prime = self.sigmoid(self.mask.unsqueeze(0) * self.temperture)
            p_prime = torch.exp(self.p_norm.unsqueeze(1).unsqueeze(0))
            z = torch.abs(x[1])
            z = torch.pow(z + self.eps, p_prime)
            z = (z + self.eps) * w_prime
            z = z.mean(dim=2, keepdim=True)
            z = torch.pow(z + self.eps, 1 / p_prime)
            z = z.reshape(z.shape[0], z.shape[1])
            z = z.unsqueeze(1)
        elif self.pooling_operation == "LSE":
            """
            Lp pooling with LSE(Log Sum Exponential)
            """
            w_prime = self.sigmoid(self.mask.unsqueeze(0) * self.temperture)
            p_prime = torch.exp(self.p_norm.unsqueeze(1).unsqueeze(0))
            z = x[1].to(torch.float64)
            z = torch.abs(z)
            t = torch.log(w_prime + self.eps) + p_prime * torch.log(z + self.eps)
            t = torch.logsumexp(t + self.eps, dim=2, keepdim=True)
            t = (t - np.log(t.shape[2])) / p_prime
            t = torch.exp(t)
            z = t.reshape(t.shape[0], t.shape[1]).unsqueeze(1)
            z = z.to(torch.float32)
        return z, w_prime

    def forward(self, x):
        y, w = self.pooling(x)
        return y, w

class lp_pooling2d(nn.Module):
    def __init__(self, device, mask, imgsz, kernel_size, stride, output_size, bias=True):
        super(lp_pooling2d, self).__init__()
        self.mask = Parameter(mask, requires_grad=True)
        self.p_norm = Parameter(torch.zeros(output_size).add_(4), requires_grad=True)

        self.eps = 1e-60
        self.sigmoid = nn.Sigmoid()
        self.temperture = 5
        self.pooling_operation = "Non_LSE"

        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=stride)
        self.fold = nn.Fold(output_size=(imgsz // kernel_size, imgsz // kernel_size), kernel_size=(1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(output_size))

        self.ondo_w = "True"
        self.exp_p = "True"

    def pooling(self, x):
        if self.ondo_w == "True":
            w_prime = self.sigmoid(self.mask.unsqueeze(0) * self.temperture)
        else:
            w_prime = self.sigmoid(self.mask.unsqueeze(0))

        if self.exp_p == "True":
            p_prime = torch.exp(self.p_norm.unsqueeze(0).unsqueeze(0))
        else:
            p_prime = self.p_norm.unsqueeze(0).unsqueeze(0)

        if self.pooling_operation == "Non_LSE":
            p_prime = p_prime.unsqueeze(0)
            w_prime = w_prime.unsqueeze(0)
            z = x[1].to(torch.float64)
            z = self.unfold(z)
            z = z.reshape(x[1].shape[0],x[1].shape[1],4,z.shape[2])
            z = torch.abs(z)
            z = torch.pow(z + self.eps, p_prime)
            z = (z + self.eps) * w_prime
            z = z.mean(dim=2, keepdim=True)
            z = torch.pow(z + self.eps, 1 / p_prime)
            z = z.reshape(z.shape[0], -1, z.shape[3])
            z = self.fold(z)
            z = z.to(torch.float32)

        elif self.pooling_operation =="LSE":
            p_prime = p_prime.unsqueeze(0)
            w_prime = w_prime.unsqueeze(0)
            z = x[1].to(torch.float64)
            z = self.unfold(z)
            z = z.reshape(x[1].shape[0],x[1].shape[1],4,z.shape[2])
            z = torch.abs(z)
            t = torch.log(w_prime + self.eps) + p_prime * torch.log(z + self.eps)
            t = torch.logsumexp(t + self.eps, dim=2, keepdim=True)
            t = (t - np.log(t.shape[2])) / p_prime
            t = torch.exp(t)
            z = t.reshape(t.shape[0], -1, t.shape[3])
            z = self.fold(z)
            z = z.to(torch.float32)
        return z, w_prime

    def forward(self, x):
        z, w = self.pooling(x)
        return z, w