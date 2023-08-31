import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

dataset_size = 60000
batch_size = 100
percetile = 95

### load numpy arrays
cifar_X = np.load("./synced_data/csv/cifar100/clip_emb_cifar100.npy")
imagenet_X = np.load("./synced_data/csv/imagenet-1k/clip_emb_imagenet-1k.npy")
X = np.concatenate((cifar_X, imagenet_X), axis=0)
arr = np.random.choice(np.arange(X.shape[0]), dataset_size, replace=True)
X = X[arr, :]
cifar_y = np.load("./synced_data/csv/cifar100/cifar100_val.npy")
imagent_y = np.load("./synced_data/csv/imagenet-1k/imagenet-1k_val.npy")
y = np.concatenate((cifar_y, imagent_y), axis=0)
y = y[arr, :]

### load cumulative reward
rewards_ucb = np.load(
    f"./synced_data/cumulative_reward/BootstrappedUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percetile}.npy"
)
rewards_egr = np.load(
    f"./synced_data/cumulative_reward/EpsilonGreedy_ds{dataset_size}_bs{batch_size}_per{percetile}.npy"
)
rewards_lucb = np.load(
    f"./synced_data/cumulative_reward/LogisticUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percetile}.npy"
)

### load optimal reward
rewards_opt = np.array(int(y.shape[0] / batch_size) * [y.max(axis=1).mean()])

opt_bandit = y.mean(axis=0).argmax()
print("Optimal bandit: ", opt_bandit)

### load PPO reward
ppo = pd.read_csv("./synced_data/cumulative_reward/step100_embed.csv")
### pandas to numpy
rewards_ppo = ppo["Value"].to_numpy()[: rewards_opt.shape[0]]
rewards_ppo /= batch_size

### load PPO reward
ppo_img = pd.read_csv("./synced_data/cumulative_reward/step100_img.csv")
### pandas to numpy
rewards_ppo_img = ppo_img["Value"].to_numpy()[: rewards_opt.shape[0]]
rewards_ppo_img /= batch_size


def get_mean_reward(reward_lst, batch_size=batch_size):
    mean_rew = list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[: r + 1]) * 1.0 / ((r + 1) * batch_size))
    return mean_rew


rcParams["figure.figsize"] = 14, 8
lwd = 5
cmap = plt.get_cmap("tab20")
colors = plt.cm.tab20(np.linspace(0, 1, 20))

ax = plt.subplot(111)
plt.plot(
    get_mean_reward(rewards_ucb),
    label=f"Bootstrapped UCB (C.I.={percetile}%)",
    linewidth=lwd,
    color=colors[0],
)
plt.plot(
    get_mean_reward(rewards_egr),
    label="$\epsilon$-Greedy",
    linewidth=lwd,
    color=colors[6],
)  ### (p0=20%, decay=0.9999) , marker='o', linestyle=':'
plt.plot(
    get_mean_reward(rewards_lucb),
    label=f"Logistic UCB (C.I.={percetile}%)",
    linewidth=lwd,
    color=colors[8],
)
plt.plot(rewards_ppo_img, label="PPO without CLIP", linewidth=lwd, color=colors[14])
plt.plot(rewards_ppo, label="PPO with CLIP", linewidth=lwd, color=colors[12])
plt.plot(
    rewards_opt,
    label="Optimal Policy",
    linewidth=lwd,
    color=colors[2],
    ls="dashed",
)
plt.plot(
    np.repeat(y.mean(axis=0).max(), len(rewards_ucb)),
    label="Overall Best Arm (no context)",
    linewidth=lwd,
    color=colors[1],
    ls="-.",
)
plt.plot(
    np.repeat(y.mean(axis=0).min(), len(rewards_ucb)),
    label="Overall Worst Arm (no context)",
    linewidth=lwd,
    color=colors[4],
    ls=":",
)

# import warnings\
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
plt.ylabel("Cumulative Mean Reward", size=25)
plt.title("Image Classification", size=30)
plt.grid()
# plt.show()
plt.savefig(
    f"./plot/image_classify_ds{dataset_size}_bs{batch_size}_per{percetile}.png",
    bbox_inches="tight",
)
