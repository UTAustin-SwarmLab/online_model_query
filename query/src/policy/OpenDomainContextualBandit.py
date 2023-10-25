### code modified from https://github.com/david-cortes/contextualbandits
### poetry run python query/src/policy/QAContextualBandit.py
# Turning logistic regression into contextual bandits policies:
import os
from copy import deepcopy

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
from contextualbandits.online import (
    BootstrappedUCB,
    EpsilonGreedy,
    LogisticUCB,
)
from pylab import rcParams
from sklearn.linear_model import LogisticRegression

bandits = {
    0: "vicuna-7b-v1.5",
    1: "falcon-180B",
    2: "falcon-180B-chat",
    3: "qCammel-70-x",
    4: "Llama-2-70b-instruct",
    5: "Llama-2-70b-instruct-v2",
    6: "StableBeluga-13B",
    7: "airoboros-l2-70b",
}
data_path = "synced_data/csv/mmlu/"

# batch size - algorithms will be refit after N rounds
batch_size = 100
dataset_size = 25000
percentile = 95
random_seed = 42
dataset = "mmlu"
max_iter = 2000
### set random seed
np.random.seed(random_seed)

### idx
model_idx = [i for i in bandits.keys()]

### load embeddings
question_np = np.load(data_path + "clip_emb_question.npy")
context_np = np.load(data_path + "clip_emb_choices.npy")
model_answer_np = np.load(data_path + "clip_emb_answer.npy")
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
input("Press Enter to continue...")

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
    f"./synced_data/cumulative_reward/BootstrappedUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy",
    rewards_ucb,
)
np.save(
    f"./synced_data/cumulative_reward/EpsilonGreedy_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy",
    rewards_egr,
)
np.save(
    f"./synced_data/cumulative_reward/LogisticUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percentile}_{dataset}.npy",
    rewards_lucb,
)

# import warnings
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
