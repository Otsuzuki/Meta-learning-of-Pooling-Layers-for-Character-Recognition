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
    Visualization W 1d
"""
def Visualize_W1d(net, step_idx, w):
    w = w.squeeze().detach().cpu().numpy()
    a = net[0].mask.detach().cpu().numpy()

    fig ,ax = plt.subplots(figsize=(10,8),dpi=300)
    sc = ax.imshow(w, cmap=cm.gray)
    plt.colorbar(sc)
    plt.savefig(os.path.join('./result/w/', 'matrix%d.jpeg' % (step_idx+1)))
    plt.close()
"""
    Visualization W 2d
"""
def Visualize_W2d(net, step_idx, w, args):
    w = w.squeeze().reshape(args.kernel_size,args.kernel_size,args.output_size).permute(2,0,1)
    outputs = []
    for i in range(args.output_size):
        if (i + 1) % (args.input_size // args.kernel_size) == 1:
            out = w[i]
        else:
            out = torch.cat([out,w[i]], dim=1)
        if (i + 1) % (args.input_size // args.kernel_size) == 0 & (i + 1) != 1:
            outputs.append(out)
            out = 0
    result = torch.cat(outputs,dim=0)

    w = w.detach().cpu().numpy()
    result = result.detach().cpu().numpy()

    if not os.path.exists('./result/w/POOLING/'):
        os.makedirs('./result/w/POOLING/')

    fig ,ax = plt.subplots(figsize=(10,8),dpi=300)
    sc = ax.imshow(result, vmin=0, vmax=1.0, cmap=cm.gray)
    plt.colorbar(sc, pad=0.2)
    plt.title("max=%f, min=%f " %(result.max(), result.min()))
    plt.savefig(os.path.join('./result/w/POOLING/' + '%d.jpg' % ((step_idx + 1) // 20 + 1)), bbox_inches='tight', dpi=300)
    plt.close()


def Visualize_W2d_color(net, step_idx, args):
    w = torch.ones(args.output_size, args.kernel_size * 2)
    # p = torch.exp(net[0].p_norm)
    p = net[0].p_norm.reshape(1,args.output_size)
    p = p.squeeze().cpu()
    outputs = []
    for i in range(args.output_size):
        if (i + 1) % (args.input_size // args.kernel_size) == 1:
            out = w[i] * p[i]
            out = out.reshape(args.kernel_size, args.kernel_size)
        else:
            x = w[i] * p[i]
            x = x.reshape(args.kernel_size, args.kernel_size)
            out = torch.cat([out,x], dim=1)
        if (i + 1) % (args.input_size // args.kernel_size) == 0 & (i + 1) != 1:
            outputs.append(out)
            out = 0

    result = torch.cat(outputs,dim=0)
    result = result.detach().cpu().numpy()
    # result[:,1::2] = -100

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import numpy as np
    if not os.path.exists('./result/w/POOLING_COLOR/'):
        os.makedirs('./result/w/POOLING_COLOR/')
    cmap = cm.coolwarm
    cmap_data = cmap(np.arange(cmap.N))
    cmap_data[0, 3] = 0 # 0 のときのα値を0(透明)にする
    customized_cool = colors.ListedColormap(cmap_data)

    plt.imshow(np.exp(result),vmin=0,vmax=120.0,cmap=customized_cool)
    plt.tick_params(bottom=False,
               left=False,
               right=False,
               top=False)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.colorbar()
    plt.savefig(os.path.join('./result/w/POOLING_COLOR/' + '%d.jpg' % ((step_idx + 1) // 20 + 1)), bbox_inches='tight', dpi=300)
    plt.close()