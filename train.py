import argparse
import configparser
import os
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
from layers import lp_pooling
from synthetic_loader import SyntheticLoader
from inner_optimizers import InnerOptBuilder
from figure import figure_loss, Visualize_W

OUTPUT_PATH = "./outputs/synthetic_outputs"


def train(device, step_idx, data, net, mask, inner_opt_builder, meta_opt, n_inner_iter):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    pred = []
    meta_opt.zero_grad()
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
            for _ in range(n_inner_iter):
                input_list = [step_idx, x_spt[i].to(device)]
                spt_pred, w = fnet(input_list)
                spt_loss = F.mse_loss(spt_pred, y_spt[i].to(device))
                diffopt.step(spt_loss)
            input_list = [step_idx, x_qry[i].to(device)]
            qry_pred, w = fnet(input_list)
            qry_loss = F.mse_loss(qry_pred, y_qry[i].to(device))
            qry_losses.append(qry_loss.detach().cpu().numpy())
            pred.append(qry_pred)
            qry_loss.backward()
    avg_qry_loss = np.mean(qry_losses)
    meta_opt.step()
    return avg_qry_loss, w

def test(device, step_idx, data, net, inner_opt_builder, n_inner_iter):
    """Main meta-test step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    for i in range(task_num):
        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                input_list = [step_idx, x_spt[i].to(device)]
                spt_pred, w = fnet(input_list)
                regularization_loss = torch.sum(torch.abs(w))
                spt_loss = F.mse_loss(spt_pred, y_spt[i].to(device))
                diffopt.step(spt_loss)
            input_list = [step_idx, x_qry[i].to(device)]
            qry_pred, w = fnet(input_list)
            qry_loss = F.mse_loss(qry_pred, y_qry[i].to(device))
            qry_losses.append(qry_loss.detach().cpu().numpy())
            y_ans = y_qry[i]
    avg_qry_loss = np.mean(qry_losses)
    _low, high = st.t.interval(
        0.95, len(qry_losses) - 1, loc=avg_qry_loss, scale=st.sem(qry_losses)
    )
    return avg_qry_loss, qry_pred, y_ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.1)
    parser.add_argument("--outer_lr", type=float, default=0.05)
    parser.add_argument("--k_spt", type=int, default=1)
    parser.add_argument("--k_qry", type=int, default=19)
    parser.add_argument("--lr_mode", type=str, default="per_layer")
    parser.add_argument("--num_inner_steps", type=int, default=1)
    parser.add_argument("--num_outer_steps", type=int, default=10000)
    parser.add_argument("--inner_opt", type=str, default="maml_adam")
    parser.add_argument("--outer_opt", type=str, default="Adam")
    parser.add_argument("--problem", type=str, default="Half_maxavg")
    parser.add_argument("--model", type=str, default="Pooling")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument('--input_size', type=int, help='input_size', default=60)
    parser.add_argument('--output_size', type=int, help='output_size', default=30)
    parser.add_argument('--kernel_size', type=int, help='kernel_size', default=2)
    parser.add_argument('--stride', type=int, help='stride', default=2)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    args = parser.parse_args()
    config = configparser.ConfigParser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db = SyntheticLoader(device, problem=args.problem, k_spt=args.k_spt, k_qry=args.k_qry)

    if args.model == "Pooling":
        mask = torch.rand(args.output_size, args.input_size)
        net = nn.Sequential(layers.lp_pooling(mask, args, bias=False)).to(device)
    else:
        raise ValueError(f"Invalid model {args.model}")

    inner_opt_builder = InnerOptBuilder(
        net, device, args.inner_opt, args.init_inner_lr, "learned", args.lr_mode
    )
    if args.outer_opt == "SGD":
        meta_opt = optim.SGD(inner_opt_builder.metaparams.values(), lr=args.outer_lr)
    else:
        meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=args.outer_lr)

    start_time = time.time()
    train_losses = []
    valid_losses = []
    for step_idx in range(args.num_outer_steps):
        data, _filters = db.next(32, "train")
        train_loss, w = train(device, step_idx, data, net, mask, inner_opt_builder, meta_opt, args.num_inner_steps)
        train_losses.append(train_loss)
        if step_idx < 60000:
            test_data, _filters  = db.next(300, "test")
            val_loss, y_pred, y_ans = test(
                device,
                step_idx,
                test_data,
                net,
                inner_opt_builder,
                args.num_inner_steps,
            )
            valid_losses.append(val_loss)
            if step_idx >= 0:
                steps_p_sec = time.time() - start_time
                print(f"Step: {step_idx+1}. Steps/sec: {steps_p_sec:.2f}")

            if step_idx == 0 or (step_idx + 1) % 20 == 0:
                Visualize_W(net, step_idx, w)

        p = net[0].p_norm.reshape(1,net[0].p_norm.shape[0])
        with open("./result/p.csv", 'a') as f:
            np.savetxt(f, torch.exp(p).detach().cpu().numpy(), delimiter=',')

    figure_loss(train_losses,valid_losses)


if __name__ == "__main__":
    if not os.path.exists('./result/w'):
        os.makedirs('./result/w')

    main()
