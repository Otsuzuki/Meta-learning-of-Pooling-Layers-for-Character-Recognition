import argparse
import configparser
import os
import random
import time
import csv
import scipy.stats as st
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import higher
import matplotlib.pyplot as plt
import layers
from loader import MiniImagenetDataset, OmniglotDataset
from layers import MetaLinearModel, MetaConvModel
from inner_optimizers_test import InnerOptBuilder_test
from figure import Visualize_W, Visualize_W_color, figure_loss, figure_acc

OUTPUT_PATH = "./outputs"

def train(device, step_idx, data, net, inner_opt_builder, meta_opt, n_inner_iter):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    qry_acces = []
    meta_opt.zero_grad()
    criterion = nn.CrossEntropyLoss()
    for i in range(task_num):
        with higher.innerloop_ctx(
            net,
            inner_opt,
            copy_initial_weights=False,
            override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            correct = 0
            total = 0
            for _ in range(n_inner_iter):
                input_list = [step_idx, x_spt[i].to(device), 0, "test"]
                spt_pred, w = fnet(input_list)
                spt_loss = criterion(spt_pred.float(), y_spt[i].long())
                diffopt.step(spt_loss)
            input_list = [step_idx, x_qry[i].to(device), 0, "test"]
            qry_pred, w = fnet(input_list)
            qry_loss = criterion(qry_pred.float(), y_qry[i].long())
            qry_losses.append(qry_loss.detach().cpu().numpy())
            qry_loss.backward()

            _, predicted = torch.max(qry_pred, 1)
            correct += (predicted == y_qry[i]).sum().item()
            total += y_qry[i].size(0)
            qry_acces.append(float(correct) / total)
    avg_qry_loss = np.mean(qry_losses)
    avg_qry_acc = np.mean(qry_acces)
    meta_opt.step()
    return avg_qry_loss, avg_qry_acc, w

def test(device, step_idx, data, net, inner_opt_builder, n_inner_iter):
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    qry_acces = []
    criterion = nn.CrossEntropyLoss()
    for i in range(task_num):
        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            correct = 0
            total = 0
            for _ in range(n_inner_iter):
                input_list = [step_idx, x_spt[i].to(device), 0, "test"]
                spt_pred, w = fnet(input_list)
                spt_loss = criterion(spt_pred.float(), y_spt[i].long())
                diffopt.step(spt_loss)
            input_list = [step_idx, x_qry[i].to(device), 0, "test"]
            qry_pred, w = fnet(input_list)
            qry_loss = criterion(qry_pred.float(), y_qry[i].long())
            qry_losses.append(qry_loss.detach().cpu().numpy())

            _, predicted = torch.max(qry_pred, 1)
            correct += (predicted == y_qry[i]).sum().item()
            total += y_qry[i].size(0)
            qry_acces.append(float(correct) / total)
    avg_qry_loss = np.mean(qry_losses)
    avg_qry_acc = np.mean(qry_acces)
    return avg_qry_loss, avg_qry_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.05)
    parser.add_argument("--outer_lr", type=float, default=0.00005)
    parser.add_argument("--k_spt", type=int, default=5)
    parser.add_argument("--k_qry", type=int, default=15)
    parser.add_argument("--lr_mode", type=str, default="per_layer")
    parser.add_argument("--num_inner_steps", type=int, default=1)
    parser.add_argument("--num_outer_steps", type=int, default=10000)
    parser.add_argument("--inner_opt", type=str, default="maml")
    parser.add_argument("--outer_opt", type=str, default="Adam")
    parser.add_argument("--problem", type=str, default="omniglot")
    parser.add_argument("--model", type=str, default="MetaConvModel")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument('--p', type=int, help='p', default=196)
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    parser.add_argument('--kernel_size', type=int, help='kernel_size', default=2)
    parser.add_argument('--stride', type=int, help='stride', default=2)
    parser.add_argument('--hidden_size', type=int, default=64, help='Number of channels in each convolution layer of the VGG network ')

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    args = parser.parse_args()
    config = configparser.ConfigParser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.problem == "omniglot":
        db = OmniglotDataset(device, problem=args.problem, task_num=args.task_num, n_way=args.n_way, imgsz=args.imgsz, k_spt=args.k_spt, k_qry=args.k_qry)
    elif args.problem == "miniimagenet":
        db = MiniImagenetDataset(device, problem=args.problem, task_num=args.task_num, n_way=args.n_way, imgsz=args.imgsz, k_spt=args.k_spt, k_qry=args.k_qry)

    if args.model == "MetaLinearModel":
        net=MetaLinearModel(args).to(device)
    elif args.model == "MetaConvModel":
        net=MetaConvModel(args).to(device)
        net.load_state_dict(torch.load('./model.pth', map_location=device))
    else:
        raise ValueError(f"Invalid model {args.model}")

    inner_opt_builder = InnerOptBuilder_test(
        net, device, args.inner_opt, args.init_inner_lr, "learned", args.lr_mode
    )
    if args.outer_opt == "SGD":
        meta_opt = optim.SGD(inner_opt_builder.metaparams.values(), lr=args.outer_lr)
    else:
        meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=args.outer_lr)

    start_time = time.time()
    train_losses = []
    train_acces = []
    valid_losses = []
    valid_acces = []
    for step_idx in range(args.num_outer_steps):
        data = db.next(32, "train")
        train_loss, train_acc, w = train(device, step_idx, data, net, inner_opt_builder, meta_opt, args.num_inner_steps)
        train_losses.append(train_loss)
        train_acces.append(train_acc)

        if step_idx < 60000:
            test_data = db.next(100, "val")
            val_loss, val_acc = test(
                device,
                step_idx,
                test_data,
                net,
                inner_opt_builder,
                args.num_inner_steps,
            )
            valid_losses.append(val_loss)
            valid_acces.append(val_acc)

            if step_idx >= 0:
                steps_p_sec = time.time() - start_time
                minite = steps_p_sec // 60
                second = steps_p_sec % 60
                hour = minite // 60
                minite = minite - 60 * hour
                print(f"Step: {step_idx+1} / {args.num_outer_steps}. Steps/sec: {hour:.2f}h {minite:.2f}m {second:.2f}s  acc: {val_acc:.5f}.  loss: {val_loss:.6f}.")

            if step_idx == 0:
                Visualize_W(net, step_idx, w, args)
                Visualize_W_color(net, step_idx, args)

    figure_loss(train_losses, valid_losses)
    figure_acc(train_acces, valid_acces)

if __name__ == "__main__":
    seed = 4
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not os.path.exists('./result/w'):
        os.makedirs('./result/w')

    main()
