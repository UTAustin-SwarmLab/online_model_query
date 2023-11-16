import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from utils import get_mean_reward

bandits = {
    0: "FILM",
    1: "HiTUT",
    2: "HLSM",
}

data_path = "synced_data/csv/alfred_data/"
model_path = "synced_data/cumulative_reward/"

# batch size - algorithms will be refit after N rounds
batch_size = 5
dataset_size = 13000
percentile = 95
random_seed = 42
dataset = "alfred"
max_iter = 2000
reward_metric = "SR"  # SR, GC+PLW
dataset += "_" + reward_metric
contextual_bandits = False

### set random seed
np.random.seed(random_seed)

### load token length
token_len = np.load(data_path + "instruct_token_length.npy")  # (13128,)
token_len = token_len.reshape(-1, 1)  # (13128, 1)
token_len = np.repeat(token_len, len(bandits), axis=1)  # (39384, 3)
token_len[:, 0] = 0
alpha = 0.05
beta = 0.005

### load embeddings
X_complete = np.load(data_path + "clip_emb.npy")
arm_results = np.load(data_path + "arm_results.npy")

if reward_metric == "GC":
    y_complete = arm_results[1, :, :]
elif reward_metric == "PLWGC":
    y_complete = arm_results[1, :, :] * (
        arm_results[3, :, :] / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
    )
elif reward_metric == "SR":
    y_complete = arm_results[0, :, :]
elif reward_metric == "PLWSR":
    y_complete = arm_results[0, :, :] * (
        arm_results[3, :, :] / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
    )
elif reward_metric == "GC+PLW":
    gc = arm_results[1, :, :]
    L_ratio = arm_results[3, :, :] / np.maximum(
        arm_results[2, :, :], arm_results[3, :, :]
    )
    L = arm_results[2, :, :]
    # y_complete = 0.5 * gc + 0.5 * gc * L_ratio - beta * token_len
    y_complete = 0.5 * gc - alpha * np.log10(L) - beta * token_len

print("X complete", X_complete.shape)
print("y complete", y_complete.shape)
arr = np.random.choice(np.arange(X_complete.shape[0]), dataset_size, replace=False)
X = X_complete[arr, :]
y = y_complete[arr, :]

assert X.shape[0] == y.shape[0], "X and y should have the same number of rows"
assert (
    X_complete.shape[0] == y_complete.shape[0]
), "X_complete and y_complete should have the same number of rows"

### calculate optimal reward
opt_ = y_complete.max(axis=1)  # shape: (dataset_size, )
opt_avg = opt_.mean()
opt_ = np.cumsum(opt_) / (np.arange(opt_.shape[0]) + 1)
print("Optimal mean reward: ", opt_avg)
print("Overall best arm: ", y_complete.mean(axis=0).argmax())
print("Best arm reward: ", y_complete.mean(axis=0).max())
print("Overall worst arm: ", y_complete.mean(axis=0).argmin())
print("Worst arm reward: ", y_complete.mean(axis=0).min())
print("arms: ", y_complete.mean(axis=0))

### load cumulative reward
if contextual_bandits:
    rewards_ucb = np.load(
        model_path
        + f"BootstrappedUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy"
    )
    rewards_egr = np.load(
        model_path
        + f"EpsilonGreedy_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy"
    )
    rewards_lucb = np.load(
        model_path
        + f"LogisticUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy"
    )

### calculate optimal reward
rewards_opt = np.array(int(y.shape[0] / batch_size) * [y.max(axis=1).mean()])
print("shape of rewards_opt: ", rewards_opt.shape)

### load PPO reward
ppo = pd.read_csv(model_path + f"alfred_{reward_metric}_step{batch_size}.csv")
### pandas to numpy
rewards_ppo = ppo["mean_reward"].to_numpy()[: rewards_opt.shape[0]]
print("shape of rewards_ppo: ", rewards_ppo.shape)

rcParams["figure.figsize"] = 14, 8
lwd = 5
cmap = plt.get_cmap("tab20")
colors = plt.cm.tab20(np.linspace(0, 1, 20))

ax = plt.subplot(111)
if contextual_bandits:
    plt.plot(
        get_mean_reward(rewards_ucb, batch_size),
        label=f"Bootstrapped UCB (C.I.={percentile}%)",
        linewidth=lwd,
        color=colors[0],
    )
    plt.plot(
        get_mean_reward(rewards_egr, batch_size),
        label="$\epsilon$-Greedy",
        linewidth=lwd,
        color=colors[6],
    )  ### (p0=20%, decay=0.9999) , marker='o', linestyle=':'
    plt.plot(
        get_mean_reward(rewards_lucb, batch_size),
        label=f"Logistic UCB (C.I.={percentile}%)",
        linewidth=lwd,
        color=colors[8],
    )

plt.plot(rewards_ppo, label="PPO", linewidth=lwd, color=colors[12])
plt.plot(
    rewards_opt,
    label="Optimal Policy",
    linewidth=lwd,
    color=colors[2],
    ls="dashed",
)
plt.plot(
    np.repeat(y.mean(axis=0).max(), len(rewards_ppo)),
    label="Overall Best Arm (no context)",
    linewidth=lwd,
    color=colors[1],
    ls="-.",
)
plt.plot(
    np.repeat(y.mean(axis=0).min(), len(rewards_ppo)),
    label="Overall Worst Arm (no context)",
    linewidth=lwd,
    color=colors[4],
    ls=":",
)

# import warnings
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 1.25])
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.27),
    fancybox=True,
    ncol=3,
    prop={"size": 20},
)

plt.tick_params(axis="both", which="major", labelsize=25)

plt.xlabel(f"Rounds (models were updated every {batch_size} rounds)", size=25)
plt.ylabel(f"Cumulative Mean {reward_metric}", size=25)
plt.title("ALFRED", size=30)
plt.grid()
ymin, ymax = (0, 0.21) if reward_metric == "GC+SLW" else (0.2, 0.51)
plt.yticks(np.arange(ymin, ymax, 0.05))
plt.savefig(
    f"./plot/Alfred/{dataset}_ds{dataset_size}_bs{batch_size}_per{percentile}.png",
    bbox_inches="tight",
)
