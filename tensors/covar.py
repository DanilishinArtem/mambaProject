import torch
import matplotlib.pyplot as plt
import numpy as np

def attn_score_plot(path: str, name: str):
    print("[INFO] start of working error_hist_plot for {}".format(name))
    slice_ = False
    t_mark = 100
    if "mamba" in name:
        attn_scores = torch.load("{}/{}.pt".format(path, name)).detach().cpu()
    else:
        attn_scores = torch.load("{}/{}.pt".format(path, name))[:,0,:,:].detach().cpu()
    attn_score_heatmap = attn_scores.mean(dim=0)
    attn_score_heatmap[attn_score_heatmap == float('-inf')] = -3
    attn_score_heatmap_min = attn_score_heatmap.min(dim=0, keepdim=True)[0]
    attn_score_heatmap = attn_score_heatmap - attn_score_heatmap_min
    mask = torch.triu(torch.ones_like(attn_score_heatmap), diagonal=1).bool()
    attn_score_heatmap[mask] = float('nan')

    mean_attn = attn_scores.mean(dim=0)
    if slice_:
        time = torch.arange(mean_attn.shape[0])
        t_grid, s_grid = torch.meshgrid(time, time, indexing='ij')
        grid = (t_grid - s_grid)[:,t_mark]
        mean_attn = mean_attn[:,t_mark]
        relative_pos = grid.flatten()
        attn_values = mean_attn.flatten()
    else:
        time = torch.arange(mean_attn.shape[0])
        t_grid, s_grid = torch.meshgrid(time, time, indexing='ij')
        relative_pos = (t_grid - s_grid).flatten()
        attn_values = mean_attn.flatten()

    attn_values[attn_values == -float('inf')] = float('inf')
    attn_values[attn_values < -10000] = float('inf')
    attn_values_min = attn_values.min(dim=0, keepdim=True)[0]
    attn_values = attn_values - attn_values_min
    # Group by each unique relative position
    from collections import defaultdict
    bins = defaultdict(list)

    for delta, val in zip(relative_pos.tolist(), attn_values.tolist()):
        if delta > 0:
            bins[delta].append(val)

    relative_pos_vals = sorted(bins.keys())
    mean_values = [sum(bins[k])/len(bins[k]) for k in relative_pos_vals]

    # Отрисовка ковариационной функции ...
    plt.figure(figsize=(10, 4))
    plt.plot(relative_pos_vals, mean_values, marker='o')
    plt.xlabel("t - s (relative position)")
    plt.ylabel("Average Attention")
    plt.title("u_s = f(t - s)")
    plt.grid(True)
    plt.savefig("{}/covar_{}.png".format(path, name))

    # Отрисовка хитмапы ...
    attn_score_heatmap = attn_score_heatmap.numpy()
    plt.figure(figsize=(10, 5))
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')
    plt.imshow(attn_score_heatmap, cmap=cmap, interpolation='none')
    plt.colorbar(label='Value')
    plt.xlabel('m')
    plt.ylabel('n')
    plt.title('Heatmap')
    plt.savefig("{}/heatmap_{}.png".format(path, name))

if __name__ == "__main__":
    path = "/home/adanilishin/mambaProject/tensors"
    # step = 10
    # step = 2000
    step = 5000
    path = "{}/step_{}".format(path, step)

    task = "copy"
    # task = "text"

    # name = "alibi_{}".format(task)
    # name = "nope_{}".format(task)
    name = "mamba_{}".format(task)

    attn_score_plot(path, name)