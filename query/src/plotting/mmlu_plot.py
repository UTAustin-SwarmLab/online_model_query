import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

bandits = {
    # 0: "vicuna-7b-v1.5",
    # 1: "falcon-180B",
    2: "falcon-180B-chat",
    # 3: "qCammel-70-x",
    4: "Llama-2-70b-instruct",
    # 5: "Llama-2-70b-instruct-v2",
    6: "StableBeluga-13B",
    # 7: "airoboros-l2-70b",
}
subset_map = json.load(open("synced_data/mmlu/subdatasets.json"))

data_path = "synced_data/csv/mmlu/"

# batch size - algorithms will be refit after N rounds
batch_size = 5
dataset_size = 10000
percentile = 95
random_seed = 42
dataset = "mmlu"
max_iter = 4000
### set random seed
np.random.seed(random_seed)

### idx
model_idx = [i for i in bandits.keys()]

### load subsets ###
subsets = pd.read_csv(data_path + "vicuna-7b-v1.5_nochoice.csv")
selected_indices = []
idx = 0
for _, row in subsets.iterrows():
    if row["subdataset"] in subset_map.values():
        selected_indices.append(idx)
    idx += 1
print(f"selected indices: {len(selected_indices)}")

### load embeddings
question_np = np.load(data_path + "clip_emb_question.npy")[selected_indices, :]
context_np = np.load(data_path + "clip_emb_choices.npy")[selected_indices, :]
model_answer_np = np.load(data_path + "clip_emb_answer.npy")[selected_indices, :]
X_complete = np.concatenate(
    (question_np, context_np, model_answer_np),
    axis=1,
)
arr = np.random.choice(np.arange(X_complete.shape[0]), dataset_size, replace=True)
# print(arr)
print("X complete", X_complete.shape)
X = X_complete[arr, :]

y_complete = np.load(data_path + "models_accnorm.npy")  # shape = [25256, 8]
print("y complete", y_complete.shape)

y_complete = y_complete[selected_indices, :]
y_complete = y_complete[:, model_idx]
y = y_complete[arr, :]

print(X.shape)
print(y.shape)
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
rewards_ucb = np.load(
    f"./synced_data/cumulative_reward/BootstrappedUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy"
)
rewards_egr = np.load(
    f"./synced_data/cumulative_reward/EpsilonGreedy_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy"
)
rewards_lucb = np.load(
    f"./synced_data/cumulative_reward/LogisticUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy"
)

### calculate optimal reward
rewards_opt = np.array(int(y.shape[0] / batch_size) * [y.max(axis=1).mean()])

### load PPO reward
ppo = pd.read_csv("./synced_data/cumulative_reward/step100_embed.csv")
### pandas to numpy
rewards_ppo = ppo["mean_reward"].to_numpy()[: rewards_opt.shape[0]]
# rewards_ppo /= batch_size**2


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
    label=f"Bootstrapped UCB (C.I.={percentile}%)",
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
plt.ylabel("Cumulative Mean Reward", size=25)
plt.title("Question Answering", size=30)
plt.grid()
# plt.show()
plt.savefig(
    f"./plot/{dataset}_ds{dataset_size}_bs{batch_size}_per{percentile}.png",
    bbox_inches="tight",
)
