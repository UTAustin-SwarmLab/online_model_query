from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from utils import get_mean_reward

bandits = {
    # 0: "deberta-v3-base-mrqa",
    1: "deberta-v3-base-squad2",
    2: "bigbird-base-trivia-itc",
    3: "distilbert-base-uncased-distilled-squad",
    4: "roberta-base-squad2-nq",
}

datasets = {
    "SQuAD": 0,
    "TriviaQA-web": 1,
    "NaturalQuestionsShort": 2,
    "NewsQA": 3,
    "SearchQA": 4,
    "HotpotQA": 5,
}

# batch size - algorithms will be refit after N rounds
batch_size = 100
dataset_size = 50000
percentile = 95
random_seed = 42
dataset = "mrqa"
max_iter = 2000
### set random seed
np.random.seed(random_seed)

### idx
model_idx = [i for i in bandits.keys()]
dataset_idx = [i for i in datasets.values()]

### load embeddings
dataset_emb_list = []
squad_question_np = np.load("./synced_data/csv/mrqa/clip_emb_SQuAD_question.npy")
squad_context_np = np.load("./synced_data/csv/mrqa/clip_emb_SQuAD_context.npy")
squad_np = np.concatenate((squad_question_np, squad_context_np), axis=1)
trivia_question_np = np.load(
    "./synced_data/csv/mrqa/clip_emb_TriviaQA-web_question.npy"
)
trivia_context_np = np.load("./synced_data/csv/mrqa/clip_emb_TriviaQA-web_context.npy")
trivia_np = np.concatenate((trivia_question_np, trivia_context_np), axis=1)
natural_question_np = np.load(
    "./synced_data/csv/mrqa/clip_emb_NaturalQuestionsShort_question.npy"
)
natural_context_np = np.load(
    "./synced_data/csv/mrqa/clip_emb_NaturalQuestionsShort_context.npy"
)
natural_np = np.concatenate((natural_question_np, natural_context_np), axis=1)
news_question_np = np.load("./synced_data/csv/mrqa/clip_emb_NewsQA_question.npy")
news_context_np = np.load("./synced_data/csv/mrqa/clip_emb_NewsQA_context.npy")
news_np = np.concatenate((news_question_np, news_context_np), axis=1)
search_question_np = np.load("./synced_data/csv/mrqa/clip_emb_SearchQA_question.npy")
search_context_np = np.load("./synced_data/csv/mrqa/clip_emb_SearchQA_context.npy")
search_np = np.concatenate((search_question_np, search_context_np), axis=1)
hotpot_question_np = np.load("./synced_data/csv/mrqa/clip_emb_HotpotQA_question.npy")
hotpot_context_np = np.load("./synced_data/csv/mrqa/clip_emb_HotpotQA_context.npy")
hotpot_np = np.concatenate((hotpot_question_np, hotpot_context_np), axis=1)

dataset_emb_list.append(squad_np)
dataset_emb_list.append(trivia_np)
dataset_emb_list.append(natural_np)
dataset_emb_list.append(news_np)
dataset_emb_list.append(search_np)
dataset_emb_list.append(hotpot_np)
dataset_emb_list = itemgetter(*dataset_idx)(dataset_emb_list)

X_complete = np.concatenate(
    (dataset_emb_list),
    axis=0,
)
arr = np.random.choice(np.arange(X_complete.shape[0]), dataset_size, replace=True)
# print(arr)
print("X complete", X_complete.shape)
X = X_complete[arr, :]

# remove SQuAD
dataset_y_list = []
squad_exact_np = np.load("./synced_data/csv/mrqa/SQuAD_exact.npy")
trivia_exact_np = np.load("./synced_data/csv/mrqa/TriviaQA-web_exact.npy")
natural_exact_np = np.load("./synced_data/csv/mrqa/NaturalQuestionsShort_exact.npy")
news_exact_np = np.load("./synced_data/csv/mrqa/NewsQA_exact.npy")
search_exact_np = np.load("./synced_data/csv/mrqa/SearchQA_exact.npy")
hotpot_exact_np = np.load("./synced_data/csv/mrqa/HotpotQA_exact.npy")
dataset_y_list.append(squad_exact_np)
dataset_y_list.append(trivia_exact_np)
dataset_y_list.append(natural_exact_np)
dataset_y_list.append(news_exact_np)
dataset_y_list.append(search_exact_np)
dataset_y_list.append(hotpot_exact_np)
dataset_y_list = itemgetter(*dataset_idx)(dataset_y_list)

y_complete = (
    np.concatenate(
        (dataset_y_list),
        axis=0,
    )
    / 100
)
print("y complete", y_complete.shape)
################# remove vmware model
y_complete = y_complete[:, model_idx]
y = y_complete[arr, :]

opt_bandit = y.mean(axis=0).argmax()
print("Optimal bandit: ", opt_bandit)

print(X.shape)
print(y.shape)
assert X.shape[0] == y.shape[0], "X and y should have the same number of rows"
assert (
    X_complete.shape[0] == y_complete.shape[0]
), "X_complete and y_complete should have the same number of columns"

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
ppo = pd.read_csv("./synced_data/cumulative_reward/QAstep100_embed.csv")
### pandas to numpy
rewards_ppo = ppo["mean_reward"].to_numpy()[: rewards_opt.shape[0]]
# rewards_ppo /= batch_size**2

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
    f"./plot/others/{dataset}_ds{dataset_size}_bs{batch_size}_per{percentile}.png",
    bbox_inches="tight",
)
