import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from utils import get_mean_reward

bandits = {
    0: "llava-v1.5-7b",
    1: "llava-v1.5-13b",
    2: "llava-v1.5-13b-lora",
}
data_path = "synced_data/csv/waymo/"

# batch size - algorithms will be refit after N rounds
batch_size = 5
dataset_size = 20000
percentile = 95
random_seed = 42
dataset = "Waymo"
alpha = 0.2
beta = 0.01 / 10
### set random seed
np.random.seed(random_seed)
contextual_bandits = False


def plot_waymo_lat(save=False, ax_=None):
    ### load embeddings
    q_emb = np.load(data_path + "clip_emb_question.npy")  # 10x768
    q_emb = np.tile(q_emb, (2000, 1))  # 20000x768
    token_len = np.load(data_path + "question_token_length.npy")  # 10
    token_len = np.tile(token_len, 2000)
    token_len = token_len.reshape(-1, 1)  # 20000x1
    token_len = np.repeat(token_len, 3, axis=1)  # 20000x3
    token_len[:, 0] = 0  # no need to pay for the token cost
    img_emb = np.load(data_path + "clip_emb_img.npy")  # 2000x768
    img_emb = np.repeat(img_emb, 10, axis=0)
    arm_results = np.load(data_path + "arm_results.npy")  # 20000x3
    model_latency = np.load(data_path + "arm_results_time.npy")  # 20000x3

    ### add image transmission time
    # model_latency[:, 1:] += 0.166 * 2
    df = pd.read_csv(data_path + "cloud-transmit-data-3.csv")
    df.iloc[:5, :] = df.iloc[dataset_size:, :]
    df = df.iloc[:dataset_size, :]
    network_latency = df["download"].values + df["upload"].values
    model_latency[:, 1] += network_latency
    model_latency[:, 2] += network_latency

    ### add acc and latency
    X_complete = np.concatenate(
        (q_emb, img_emb, arm_results),
        axis=1,
    )
    arr = np.random.choice(np.arange(X_complete.shape[0]), dataset_size, replace=True)
    print("X complete", X_complete.shape)
    X = X_complete[arr, :]
    y_complete = arm_results - alpha * np.log10(model_latency) - beta * token_len * 2
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
    ppo = pd.read_csv(
        f"./synced_data/cumulative_reward/waymo_latency_step{batch_size}.csv"
    )
    ### pandas to numpy
    rewards_ppo = ppo["mean_reward"].to_numpy()[: rewards_opt.shape[0]]
    steps = ppo["Step"].to_numpy()[: rewards_opt.shape[0]]
    rcParams["figure.figsize"] = 14, 8
    lwd = 5
    cmap = plt.get_cmap("tab20")
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    ax = plt.subplot(111) if ax_ is None else ax_
    if contextual_bandits:
        ax.plot(
            steps,
            get_mean_reward(rewards_ucb, batch_size),
            label=f"Bootstrapped UCB (C.I.={percentile}%)",
            linewidth=lwd,
            color=colors[0],
        )
        ax.plot(
            steps,
            get_mean_reward(rewards_egr, batch_size),
            label="$\epsilon$-Greedy",
            linewidth=lwd,
            color=colors[6],
        )  ### (p0=20%, decay=0.9999) , marker='o', linestyle=':'
        ax.plot(
            steps,
            get_mean_reward(rewards_lucb, batch_size),
            label=f"Logistic UCB (C.I.={percentile}%)",
            linewidth=lwd,
            color=colors[8],
        )
    ax.plot(steps, rewards_ppo, label="PPO (ours)", linewidth=lwd, color=colors[12])
    ax.plot(
        steps,
        rewards_opt,
        label="Optimal Policy",
        linewidth=lwd,
        color=colors[2],
        ls="dashed",
    )
    ax.plot(
        steps,
        np.repeat(y.mean(axis=0).max(), len(rewards_ppo)),
        label="Overall Best Arm (no context)",
        linewidth=lwd,
        color=colors[1],
        ls="-.",
    )
    ax.plot(
        steps,
        np.repeat(y.mean(axis=0).min(), len(rewards_ppo)),
        label="Overall Worst Arm (no context)",
        linewidth=lwd,
        color=colors[4],
        ls=":",
    )

    if save:
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 1.25]
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.27),
            fancybox=True,
            ncol=3,
            prop={"size": 20},
        )

        ax.tick_params(axis="both", which="major", labelsize=25)
        ax.grid()
        ax.set_xlabel("Steps", size=25)
        ax.set_ylabel("Cumulative Mean Success Rate", size=25)
        ax.set_title("Waymo", size=30)
        ax.set_ylim(0.6, 0.86)
        ax.set_yticks(np.arange(0.6, 0.86, 0.05))
        plt.savefig(
            f"./plot/{dataset}/{dataset}_latency_ds{dataset_size}_bs{batch_size}_per{percentile}.png",
            bbox_inches="tight",
        )
        return
    else:
        return ax


if __name__ == "__main__":
    plot_waymo_lat(save=True)
