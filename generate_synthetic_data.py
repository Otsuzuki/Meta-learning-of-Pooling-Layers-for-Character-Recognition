"""Generate synthetic data for experiments."""

import argparse
import os
# from e2cnn import gspaces
# from e2cnn import nn as gnn
# from scipy.special import softmax
import numpy as np
import torch
from torch import nn

def Max(out_path):
    m = nn.MaxPool1d(2, stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.rand(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.randn(20, 1, 60).astype(np.float32)
        result = m(torch.from_numpy(inp))
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def Average(out_path):
    m = nn.AvgPool1d(2, stride=2)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.randn(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        task_xs, task_ys = [], []
        inp = np.random.randn(20, 1, 60).astype(np.float32)
        result = m(torch.from_numpy(inp))
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def Half(out_path):
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

TYPE_2_PATH = {
    "Max": "./data/Max.npz",
    "Average": "./data/Average.npz",
    "Half_maxavg": "./data/Half_maxavg.npz",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="Average")
    args = parser.parse_args()
    out_path = TYPE_2_PATH[args.problem]

    if os.path.exists(out_path):
        raise ValueError(f"File exists at {out_path}.")
    if args.problem == "Max":
        Max(out_path)
    elif args.problem == "Average":
        Average(out_path)
    elif args.problem == "Half_maxavg":
        Half(out_path)
    else:
        raise ValueError(f"Unrecognized problem {args.problem}")


if __name__ == "__main__":
    main()

