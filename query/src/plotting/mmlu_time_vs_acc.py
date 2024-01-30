import matplotlib.pyplot as plt
import seaborn as sns
from swarm_visualizer.utility.general_utils import (
    set_axis_infos,
)

bandits = {
    0: "vicuna-7b-v1.5",
    6: "StableBeluga-13B",
    4: "Llama-2-70b-instruct",
    2: "falcon-180B-chat",
}
names = {
    0: "Vicuna-7B",
    6: "StableBeluga-13B",
    4: "LLaMA-2-70B",
    2: "Falcon-180B",
}
xylabelsize = 32
ticksize = 28
markersize = 300

sns.set(style="ticks")
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)

time = []
colors = sns.color_palette("tab10")

plt.scatter(0.04214, 0.5986, label=names[0], color=colors[6], s=markersize)
ax.annotate(
    names[0],
    xy=(0.04214, 0.5986),
    xytext=(0.04214, 0.5986 + 0.01),
    color=colors[6],
    fontsize=xylabelsize,
)
plt.scatter(0.1501, 0.6708, label=names[6], color=colors[1], s=markersize)
ax.annotate(
    names[6],
    xy=(0.1501, 0.6708),
    xytext=(0.1501 - 0.03, 0.6708 + 0.01),
    color=colors[1],
    fontsize=xylabelsize,
)
plt.scatter(0.1728, 0.7706, label=names[4], color="black", s=markersize)
ax.annotate(
    names[4],
    xy=(0.1728, 0.7706),
    xytext=(0.1728 - 0.045, 0.7706 - 0.002),
    color="black",
    fontsize=xylabelsize,
)
plt.scatter(0.2010, 0.7600, label=names[2], color=colors[2], s=markersize)
ax.annotate(
    names[2],
    xy=(0.2010, 0.7600),
    xytext=(0.2010 - 0.032, 0.7600 - 0.02),
    color=colors[2],
    fontsize=xylabelsize,
)
plt.scatter(0.1655, 0.7553, color=colors[3], label="$\epsilon$-Greedy", s=markersize)
ax.annotate(
    "$\epsilon$-Greedy",
    xy=(0.1655, 0.7553),
    xytext=(0.1655 - 0.034, 0.7553 - 0.01),
    color=colors[3],
    fontsize=xylabelsize,
)
plt.scatter(0.1016, 0.7327, color=colors[0], label="PPO (ours)", s=markersize)
ax.annotate(
    "PPO (ours)",
    xy=(0.10163234548778749, 0.7327),
    xytext=(0.10163234548778749 - 0.01, 0.7327 + 0.01),
    color=colors[0],
    fontsize=xylabelsize,
)


set_axis_infos(
    ax=ax,
    xlabel="Weighted sum of latency and costs",  # (lower is better)",
    ylabel="Accuracy",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    ticks_size=ticksize,
    xticks=[0.05, 0.1, 0.15, 0.2],
    yticks=[0.6, 0.65, 0.7, 0.75],
)
plt.grid()
plt.savefig("plot/mmlu/mmlu_time_vs_acc.png", bbox_inches="tight")
