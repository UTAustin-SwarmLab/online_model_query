import os

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams


def get_mean_reward(reward_lst, batch_size):
    mean_rew = list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[: r + 1]) * 1.0 / ((r + 1) * batch_size))
    return mean_rew


bandits = {
    0: "FILM",
    1: "HiTUT",
    2: "HLSM",
}

data_path = "synced_data/csv/alfred_data/"
model_path = "synced_data/cumulative_reward/"

# batch size - algorithms will be refit after N rounds
batch_size = 50
dataset_size = 13000
percentile = 95
random_seed = 42
dataset = "alfred"
max_iter = 2000
reward_metric = "SR"  # PLWGC, SR, PLWSR
dataset += "_" + reward_metric

### set random seed
np.random.seed(random_seed)

### load embeddings
if not (
    os.path.isfile(data_path + "clip_emb.npy")
    or os.path.isfile(data_path + "arm_results.npy")
):
    ### load embeddings
    instruction_dict = cloudpickle.load(open(data_path + "clip_emb_instruct.pkl", "rb"))
    low_level_instruction_dict = cloudpickle.load(
        open(data_path + "clip_emb_low_instruct.pkl", "rb")
    )
    floorpan_dict = cloudpickle.load(open(data_path + "floor_plan.pkl", "rb"))

    ### load csv data
    alfred_data = pd.read_csv(data_path + "alfred_merged_valid_language_goal.csv")
    arm_results_df = pd.read_csv(data_path + "alfred_models_results.csv")

    emb = []
    y = np.zeros((4, len(alfred_data), len(bandits)), dtype=np.float32)
    for _, row in alfred_data.iterrows():
        task_id = row["task_idx"]
        repeat_idx = row["repeat_idx"]
        floorplan = row["task_floor"]
        emb.append(
            np.concatenate(
                (
                    instruction_dict[(task_id, repeat_idx)],
                    low_level_instruction_dict[(task_id, repeat_idx)],
                    floorpan_dict[floorplan],
                )
            )
        )
        for i, model in bandits.items():
            result_row = arm_results_df.loc[
                (arm_results_df["task_idx"] == task_id)
                & (arm_results_df["repeat_idx"] == repeat_idx % 10)
                & (arm_results_df["model"] == model)
            ]
            sr = result_row["SR"].iloc[0]
            gc = result_row["GC"].iloc[0]
            L = result_row["L"].iloc[0]
            L_demo = result_row["L*"].iloc[0]
            y[0, _, i] = sr
            y[1, _, i] = gc
            y[2, _, i] = L
            y[3, _, i] = L_demo

    X_complete = np.array(emb)
    arm_results = np.array(y)
    np.save(data_path + "clip_emb.npy", X_complete)
    np.save(data_path + "arm_results.npy", arm_results)
else:
    X_complete = np.load(data_path + "clip_emb.npy")
    arm_results = np.load(data_path + "arm_results.npy")
    print("loaded data")

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
elif reward_metric == "gc and plw":
    gc = arm_results[1, :, :]
    plw = arm_results[1, :, :] * (
        arm_results[3, :, :] / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
    )
    y_complete = 0.5 * gc + 0.5 * plw

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

### load PPO reward
ppo = pd.read_csv(model_path + "alfred_step50_SR.csv")
### pandas to numpy
rewards_ppo = ppo["mean_reward"].to_numpy()[: rewards_opt.shape[0]]

rcParams["figure.figsize"] = 14, 8
lwd = 5
cmap = plt.get_cmap("tab20")
colors = plt.cm.tab20(np.linspace(0, 1, 20))

ax = plt.subplot(111)
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
