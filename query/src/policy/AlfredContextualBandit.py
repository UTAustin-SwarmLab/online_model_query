### code modified from https://github.com/david-cortes/contextualbandits
### poetry run python query/src/policy/QAContextualBandit.py
# Turning logistic regression into contextual bandits policies:
import os
from copy import deepcopy

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from contextualbandits.online import (
    BootstrappedUCB,
    EpsilonGreedy,
    LogisticUCB,
)
from pylab import rcParams
from sklearn.linear_model import LogisticRegression

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

X_complete = np.delete(X_complete, np.s_[768 : 768 * 4], axis=1)
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
# input("Press Enter to continue...")

### save optimal reward
os.makedirs("./cumulative_reward", exist_ok=True)

nchoices = y.shape[1]
base_algorithm = LogisticRegression(solver="lbfgs", max_iter=max_iter, warm_start=True)
beta_prior = (
    (3.0 / nchoices, 4),
    2,
)  # until there are at least 2 observations of each class, will use this prior
beta_prior_ucb = (
    (5.0 / nchoices, 4),
    2,
)  # UCB gives higher numbers, thus the higher positive prior
beta_prior_ts = ((2.0 / np.log2(nchoices), 4), 2)
### Important!!! the default values for beta_prior will be changed in version 0.3

## The base algorithm is embedded in different metaheuristics
bootstrapped_ucb = BootstrappedUCB(
    deepcopy(base_algorithm),
    nchoices=nchoices,
    beta_prior=beta_prior_ucb,
    percentile=percentile,
    random_state=random_seed,
)
epsilon_greedy = EpsilonGreedy(
    deepcopy(base_algorithm),
    nchoices=nchoices,
    beta_prior=beta_prior,
    random_state=random_seed,
)
logistic_ucb = LogisticUCB(
    nchoices=nchoices,
    percentile=percentile,
    beta_prior=beta_prior_ts,
    random_state=random_seed,
)

models = [bootstrapped_ucb, epsilon_greedy, logistic_ucb]

# These lists will keep track of the rewards obtained by each policy
rewards_ucb, rewards_egr, rewards_lucb = [list() for i in range(len(models))]

lst_rewards = [rewards_ucb, rewards_egr, rewards_lucb]

# initial seed - all policies start with the same small random selection of observation
first_batch = X[:batch_size, :]
np.random.seed(1)
action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

# fitting models for the first time
for model in models:
    model.fit(X=first_batch, a=action_chosen, r=rewards_received)

# these lists will keep track of which actions does each policy choose
lst_a_ucb, lst_a_egr, lst_a_lucb = [action_chosen.copy() for i in range(len(models))]

lst_actions = [lst_a_ucb, lst_a_egr, lst_a_lucb]


# rounds are simulated from the full dataset
def simulate_rounds(
    model, rewards, actions_hist, X_global, y_global, batch_st, batch_end
):
    np.random.seed(batch_st)

    ## choosing actions for this batch
    actions_this_batch = model.predict(X_global[batch_st:batch_end, :]).astype("uint8")

    # keeping track of the sum of rewards received
    rewards.append(y_global[np.arange(batch_st, batch_end), actions_this_batch].sum())

    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)

    # now refitting the algorithms after observing these new rewards
    np.random.seed(batch_st)
    model.fit(
        X_global[:batch_end, :],
        new_actions_hist,
        y_global[np.arange(batch_end), new_actions_hist],
    )

    return new_actions_hist


# now running all the simulation
for i in range(int(np.floor(X.shape[0] / batch_size))):
    batch_st = (i + 1) * batch_size
    batch_end = (i + 2) * batch_size
    batch_end = np.min([batch_end, X.shape[0]])
    if batch_st == batch_end:
        break
    for model in range(len(models)):
        lst_actions[model] = simulate_rounds(
            models[model],
            lst_rewards[model],
            lst_actions[model],
            X,
            y,
            batch_st,
            batch_end,
        )

### save models
with open(
    f"./synced_data/models/ucb_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.pkl",
    "wb",
) as f:
    cloudpickle.dump(models[0], f)
with open(
    f"./synced_data/models/egr_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.pkl",
    "wb",
) as f:
    cloudpickle.dump(models[1], f)
with open(
    f"./synced_data/models/lucb_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.pkl",
    "wb",
) as f:
    cloudpickle.dump(models[2], f)
with open(
    f"./synced_data/models/lst_actions_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.pkl",
    "wb",
) as f:
    cloudpickle.dump(lst_actions, f)


def get_mean_reward(reward_lst, batch_size=batch_size):
    mean_rew = list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[: r + 1]) * 1.0 / ((r + 1) * batch_size))
    return mean_rew


rcParams["figure.figsize"] = 15, 10
lwd = 5
cmap = plt.get_cmap("tab20")
colors = plt.cm.tab20(np.linspace(0, 1, 20))

ax = plt.subplot(111)
plt.plot(
    get_mean_reward(rewards_ucb),
    label=f"Bootstrapped Upper Confidence Bound (C.I.={percentile}%)",
    linewidth=lwd,
    color=colors[0],
)
plt.plot(
    get_mean_reward(rewards_egr),
    label="Epsilon-Greedy (p0=20%, decay=0.9999)",
    linewidth=lwd,
    color=colors[6],
)
plt.plot(
    get_mean_reward(rewards_lucb),
    label=f"Logistic Upper Confidence Bound (C.I.={percentile}%)",
    linewidth=lwd,
    color=colors[8],
)
plt.plot(
    np.repeat(y_complete.mean(axis=0).max(), len(rewards_ucb)),
    label="Overall Best Arm (no context)",
    linewidth=lwd,
    color=colors[1],
    ls="dashed",
)
print("Overall best arm: ", y_complete.mean(axis=0))

### save cumulative reward
np.save(
    model_path
    + f"BootstrappedUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy",
    rewards_ucb,
)
np.save(
    model_path
    + f"EpsilonGreedy_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy",
    rewards_egr,
)
np.save(
    model_path
    + f"LogisticUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy",
    rewards_lucb,
)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 1.25])
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    fancybox=True,
    ncol=2,
    prop={"size": 20},
)

plt.tick_params(axis="both", which="major", labelsize=25)

plt.xlabel(f"Rounds (models were updated every {batch_size} rounds)", size=30)
plt.ylabel("Cumulative Mean Reward", size=30)
plt.title("Comparison of Online Contextual Bandit Policies)", size=30)
plt.grid()
# plt.show()
plt.savefig(
    f"./plot/ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.png",
    bbox_inches="tight",
)
