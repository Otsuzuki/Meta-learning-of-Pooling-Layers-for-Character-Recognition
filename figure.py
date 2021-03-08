import os
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

"""
    Visualization loss
"""
def figure_loss(train_losses, valid_losses):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1,len(valid_losses)+1), valid_losses, label='Test Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./result/loss.png')
    plt.close()

"""
    Visualization test accuracy
"""
def Visualize_W(net, step_idx, w):
    w = w.squeeze().detach().cpu().numpy()
    a = net[0].mask.detach().cpu().numpy()

    fig ,ax = plt.subplots(figsize=(10,8),dpi=300)
    sc = ax.imshow(w, cmap=cm.gray)
    plt.colorbar(sc)
    plt.savefig(os.path.join('./result/w/', 'matrix%d.jpeg' % (step_idx+1)))
    plt.close()