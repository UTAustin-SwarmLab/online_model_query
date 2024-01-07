import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams
from utils import get_mean_reward

bandits = {
    0: "small",
    1: "base",
}
data_path = "synced_data/csv/rtx/"

# batch size - algorithms will be refit after N rounds
batch_size = 5
dataset_size = 15000
percentile = 95
random_seed = 42
dataset = "RTX"
alpha = 0.2
beta = 0.01
### set random seed
np.random.seed(random_seed)
contextual_bandits = False


def plot_rtx_lat(save=False, ax_=None):
    ### load embeddings
    bridge = np.load(data_path + "bridge_instruct_emb.npy")
    kuka = np.load(data_path + "kuka_instruct_emb.npy")
    fractal = np.load(data_path + "fractal20220817_data_instruct_emb.npy")
    q_emb = np.concatenate((bridge, kuka, fractal), axis=0)  # 15000 x 768

    bridge = np.load(data_path + "bridge_img_emb.npy")
    kuka = np.load(data_path + "kuka_img_emb.npy")
    fractal = np.load(data_path + "fractal20220817_data_img_emb.npy")
    img_emb = np.concatenate((bridge, kuka, fractal), axis=0)  # 15000 x 1536

    small_bridge = np.load("synced_data/rtx/bridge_small_action_errors.npy")
    small_kuka = np.load("synced_data/rtx/kuka_small_action_errors.npy")
    small_fractal = np.load(
        "synced_data/rtx/fractal20220817_data_small_action_errors.npy"
    )
    base_bridge = np.load("synced_data/rtx/bridge_base_action_errors.npy")
    base_kuka = np.load("synced_data/rtx/kuka_base_action_errors.npy")
    base_fractal = np.load(
        "synced_data/rtx/fractal20220817_data_base_action_errors.npy"
    )
    base = np.concatenate((base_bridge, base_kuka, base_fractal), axis=0)
    small = np.concatenate((small_bridge, small_kuka, small_fractal), axis=0)
    arm_results = np.stack((small, base), axis=1)  # 1500 x 2
    # make it negative (cost to reward) then scale it
    arm_results *= -20

    ### load inference time
    small_bridge = np.load("synced_data/rtx/bridge_small_times.npy")
    small_kuka = np.load("synced_data/rtx/kuka_small_times.npy")
    small_fractal = np.load("synced_data/rtx/fractal20220817_data_small_times.npy")
    base_bridge = np.load("synced_data/rtx/bridge_base_times.npy")
    base_kuka = np.load("synced_data/rtx/kuka_base_times.npy")
    base_fractal = np.load("synced_data/rtx/fractal20220817_data_base_times.npy")
    small = np.concatenate((small_bridge, small_kuka, small_fractal), axis=0)
    base = np.concatenate((base_bridge, base_kuka, base_fractal), axis=0)
    model_latency = np.stack((small, base), axis=1)  # 1500 x 2

    ### load token cost
    bridge = np.load(data_path + "bridge_instruct_length.npy")
    kuka = np.load(data_path + "kuka_instruct_length.npy")
    fractal = np.load(data_path + "fractal20220817_data_instruct_length.npy")
    token_len = np.concatenate((bridge, kuka, fractal), axis=0)  # 15000,
    token_len = np.repeat(token_len[..., np.newaxis], len(bandits), axis=1)
    token_len[:, 0] = 0  # no need to pay for the token cost

    ### add acc and latency
    X_complete = np.concatenate(
        (q_emb, img_emb),
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
        f"./synced_data/cumulative_reward/rtx_latency_step{batch_size}.csv"
    )
    ### pandas to numpy
    rewards_ppo = ppo["mean_reward"].to_numpy()[: rewards_opt.shape[0]]
    steps = ppo["Step"].to_numpy()[: rewards_opt.shape[0]]
    greedy = pd.read_csv(
        f"./synced_data/cumulative_reward/RTXLatency_greedy_step{batch_size}.csv"
    )
    reward_greedy = greedy["mean_reward"].to_numpy()[: rewards_opt.shape[0]]
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
        steps, reward_greedy, label="$\epsilon$-Greedy", linewidth=lwd, color=colors[18]
    )
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
        ax.set_ylabel(
            "Cumulative Negative Mean Action Error\n with Latency and Costs", size=25
        )
        ax.set_title("rtx", size=30)
        ax.set_ylim(-1.51, -1.14)
        ax.set_yticks([-1.5, -1.4, -1.3, -1.2])
        plt.savefig(
            f"./plot/{dataset}/{dataset}_latency_ds{dataset_size}_bs{batch_size}_per{percentile}.png",
            bbox_inches="tight",
        )
        return
    else:
        return ax


if __name__ == "__main__":
    plot_rtx_lat(save=True)
