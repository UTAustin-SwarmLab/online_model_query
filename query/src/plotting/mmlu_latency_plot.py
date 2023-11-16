import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from utils import get_mean_reward

bandits = {
    0: "vicuna-7b-v1.5",
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
dataset_size = 12000
percentile = 95
random_seed = 42
dataset = "mmlu"
alpha = 0.03
beta = 0.0008
### set random seed
np.random.seed(random_seed)
contextual_bandits = False

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
print("X complete", X_complete.shape)
X = X_complete[arr, :]

arm_results = np.load(data_path + "models_accnorm.npy")  # shape = [25256, 8]
arm_results = arm_results[:, model_idx]

### load latency data
model_latency = np.zeros_like(arm_results)
idx = 0
for key, value in bandits.items():
    latency = pd.read_csv(f"synced_data/csv/mmlu/{value}_answer_time.csv")
    latency = np.array(latency["answer_time"] + latency["load_time"])
    repeat_cnt = arm_results.shape[0] // len(latency) + 1
    latency = np.tile(latency, repeat_cnt)
    model_latency[:, idx] = latency[selected_indices]
    idx += 1

### load token costs
token_len = np.zeros_like(arm_results)
token_length = np.load("synced_data/csv/mmlu/question_token_length.npy")
token_len[:, 1:] = token_length[selected_indices, np.newaxis]

### add acc and latency
y_complete = arm_results - alpha * np.log10(model_latency) - beta * token_len

y_complete = y_complete[selected_indices, :]
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

if contextual_bandits:
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
ppo = pd.read_csv(f"./synced_data/cumulative_reward/mmlu_latency_step{batch_size}.csv")
### pandas to numpy
rewards_ppo = ppo["mean_reward"].to_numpy()[: rewards_opt.shape[0]]
print("PPO mean reward: ", rewards_ppo.shape)
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
plt.ylabel("Cumulative Mean Reward", size=25)
plt.title("Question Answering", size=30)
plt.grid()
plt.savefig(
    f"./plot/mmlu/{dataset}_latency_ds{dataset_size}_bs{batch_size}_per{percentile}.png",
    bbox_inches="tight",
)
