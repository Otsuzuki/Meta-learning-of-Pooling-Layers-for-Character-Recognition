"""Generate synthetic data for experiments."""

import argparse
import os
# from e2cnn import gspaces
# from e2cnn import nn as gnn
# from scipy.special import softmax
import numpy as np
import torch
from torch import nn

def Max_1d(out_path):
    m = nn.MaxPool1d(2, stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.rand(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.rand(20, 1, 60).astype(np.float32)
        result = m(torch.from_numpy(inp))
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def Average_1d(out_path):
    m = nn.AvgPool1d(2, stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.rand(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.rand(20, 1, 60).astype(np.float32)
        result = m(torch.from_numpy(inp))
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def Square_1d(out_path):
    m = nn.MaxPool1d(2, stride=2)
    a = nn.AvgPool1d(2, stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.rand(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.rand(20, 1, 60).astype(np.float32)
        inp = torch.from_numpy(inp)
        x = inp[:,:,:30]
        x = m(x)
        xx = inp[:,:,30:]
        xx = a(xx)
        result = torch.cat((x, xx), dim=2)
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def Max_2d(out_path):
    m = nn.MaxPool2d(2, stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.rand(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.rand(20, 1, 28, 28).astype(np.float32)
        result = m(torch.from_numpy(inp))
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def Average_2d(out_path):
    m = nn.AvgPool2d(2, stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.rand(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.rand(20, 1, 28, 28).astype(np.float32)
        result = m(torch.from_numpy(inp))
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def Square_2d(out_path):
    m = nn.MaxPool2d((2,2), stride=2)
    a = nn.AvgPool2d((2,2), stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.rand(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.rand(20, 1, 28, 28).astype(np.float32)
        inp = torch.from_numpy(inp)
        x = inp[:,:,:14,:]
        x = m(x)
        xx = inp[:,:,14:,:]
        xx = a(xx)
        result = torch.cat((x, xx), dim=2)
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def NonSquare_2d(out_path):
    m = nn.MaxPool2d((2,1), stride=2)
    a = nn.AvgPool2d((2,1), stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.rand(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.rand(20, 1, 28, 28).astype(np.float32)
        inp = torch.from_numpy(inp)
        x = inp[:,:,:14,:]
        x = m(x)
        xx = inp[:,:,14:,:]
        xx = a(xx)
        result = torch.cat((x, xx), dim=2)
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)



TYPE_2_PATH = {
    "Max_1d": "./data/Max_1d.npz",
    "Average_1d": "./data/Average_1d.npz",
    "Square_1d": "./data/Square_1d.npz",
    "Max_2d": "./data/Max_2d.npz",
    "Average_2d": "./data/Average_2d.npz",
    "Square_2d": "./data/Square_2d.npz",
    "NonSquare_2d": "./data/NonSquare_2d.npz",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="Max_2d")
    args = parser.parse_args()
    out_path = TYPE_2_PATH[args.problem]

    if os.path.exists(out_path):
        raise ValueError(f"File exists at {out_path}.")
    if args.problem == "Max_1d":
        Max_1d(out_path)
    elif args.problem == "Average_1d":
        Average_1d(out_path)
    elif args.problem == "Square_1d":
        Square_1d(out_path)
    elif args.problem == "Max_2d":
        Max_2d(out_path)
    elif args.problem == "Average_2d":
        Average_2d(out_path)
    elif args.problem == "Square_2d":
        Square_2d(out_path)
    elif args.problem == "NonSquare_2d":
        NonSquare_2d(out_path)
    else:
        raise ValueError(f"Unrecognized problem {args.problem}")


if __name__ == "__main__":
    main()

