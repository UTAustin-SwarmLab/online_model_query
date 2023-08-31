### code modified from https://github.com/david-cortes/contextualbandits
# Turning logistic regression into contextual bandits policies:
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

# batch size - algorithms will be refit after N rounds
batch_size = 100
dataset_size = 60000
percetile = 95
random_seed = 42

### set random seed
np.random.seed(random_seed)

### load numpy arrays
cifar_X = np.load("./csv/cifar100/clip_emb_cifar100.npy")
imagenet_X = np.load("./csv/imagenet-1k/clip_emb_imagenet-1k.npy")
X = np.concatenate((cifar_X, imagenet_X), axis=0)
arr = np.random.choice(np.arange(X.shape[0]), dataset_size, replace=True)
print(arr)
X = X[arr, :]

cifar_y = np.load("./csv/cifar100/cifar100_val.npy")
imagent_y = np.load("./csv/imagenet-1k/imagenet-1k_val.npy")
y = np.concatenate((cifar_y, imagent_y), axis=0)
y = y[arr, :]

print(X.shape)
print(y.shape)

### calculate optimal reward
opt_ = y.max(axis=1)  # shape: (dataset_size, )
opt_ = np.cumsum(opt_) / (np.arange(opt_.shape[0]) + 1)
### save optimal reward
np.save(f"./cumulative_reward/OptimalReward_ds{dataset_size}.npy", opt_)

nchoices = y.shape[1]
base_algorithm = LogisticRegression(solver="lbfgs", max_iter=1000, warm_start=True)
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
    percentile=percetile,
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
    percentile=percetile,
    beta_prior=beta_prior_ts,
    random_state=random_seed,
)

models = [bootstrapped_ucb, epsilon_greedy, logistic_ucb]

# These lists will keep track of the rewards obtained by each policy
rewards_ucb, rewards_egr, rewards_lucb = [list() for i in range(len(models))]

lst_rewards = [rewards_ucb, rewards_egr, rewards_lucb]

# initial seed - all policies start with the same small random selection of actions/rewards
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
    f"./models/ucb_ds{dataset_size}_bs{batch_size}_per{percetile}.pkl", "wb"
) as f:
    cloudpickle.dump(models[0], f)
with open(
    f"./models/egr_ds{dataset_size}_bs{batch_size}_per{percetile}.pkl", "wb"
) as f:
    cloudpickle.dump(models[1], f)
with open(
    f"./models/lucb_ds{dataset_size}_bs{batch_size}_per{percetile}.pkl", "wb"
) as f:
    cloudpickle.dump(models[2], f)

with open(
    f"./models/lst_actions_ds{dataset_size}_bs{batch_size}_per{percetile}.pkl", "wb"
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
    label=f"Bootstrapped Upper Confidence Bound (C.I.={percetile}%)",
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
    label=f"Logistic Upper Confidence Bound (C.I.={percetile}%)",
    linewidth=lwd,
    color=colors[8],
)
plt.plot(
    np.repeat(y.mean(axis=0).max(), len(rewards_ucb)),
    label="Overall Best Arm (no context)",
    linewidth=lwd,
    color=colors[1],
    ls="dashed",
)

### save cumulative reward
np.save(
    f"./cumulative_reward/BootstrappedUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percetile}.npy",
    rewards_ucb,
)
np.save(
    f"./cumulative_reward/EpsilonGreedy_ds{dataset_size}_bs{batch_size}_per{percetile}.npy",
    rewards_egr,
)
np.save(
    f"./cumulative_reward/LogisticUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percetile}.npy",
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
# plt.xticks([i*batch_size for i in range(int(np.floor(X.shape[0] / batch_size)))], [i*batch_size for i in range(int(np.floor(X.shape[0] / batch_size)))])

plt.xlabel(f"Rounds (models were updated every {batch_size} rounds)", size=30)
plt.ylabel("Cumulative Mean Reward", size=30)
plt.title("Comparison of Online Contextual Bandit Policies)", size=30)
plt.grid()
# plt.show()
plt.savefig(
    f"./plot/BootstrappedUpperConfidenceBound_EpsilonGreedy_LogisticUpperConfidenceBound_ds{dataset_size}_bs{batch_size}_per{percetile}.png",
    bbox_inches="tight",
)
