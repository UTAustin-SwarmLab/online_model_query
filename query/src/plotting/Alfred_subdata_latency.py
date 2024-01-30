import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from swarm_visualizer.barplot import (
    plot_sns_grouped_barplot,  # noqa F401
)
from swarm_visualizer.utility.general_utils import (
    set_axis_infos,
)

colors = sns.color_palette("tab10", 7)
xylabelsize = 32
legendsize = 28
ticksize = 32
alpha = 0.05
beta = 0.005
bandits = {
    0: "FILM",
    1: "HiTUT",
    2: "HLSM",
}
data_path = "synced_data/csv/alfred_data/"

token_len = np.load(data_path + "instruct_token_length.npy")  # (13128,)
token_len = token_len.reshape(13128)[:1641]
df = pd.read_csv("synced_data/csv/alfred_data/alfred_models_results.csv")
df.sort_values(by=["task_type"], inplace=True)
for model in bandits.values():
    if model == "FILM":
        df.loc[df["model"] == model, "token_len"] = 0
    else:
        df.loc[df["model"] == model, "token_len"] = token_len
df["reward_latency"] = (
    0.5 * df["GC"] - alpha * np.log10(df["L"]) - beta * df["token_len"]
)

fig, ax = plt.subplots(figsize=(18, 8))
plt.grid()
sns.barplot(
    ax=ax,
    x="model",
    y="reward_latency",
    palette=colors,
    hue="task_type",
    data=df,
    hue_order=[
        "pick_clean_then_place_in_recep",
        "look_at_obj_in_light",
        "pick_and_place_simple",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
        "pick_and_place_with_movable_recep",
    ],
)
ax.get_legend().remove()

plt.subplots_adjust(hspace=0.12)
set_axis_infos(
    ax,
    ylabel="Mean Rewards of Models\nwith Latency and Costs",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    xlabel="Models",
    ticks_size=ticksize,
)
# ax.set_yticks([0.0, 0.1])

lines_labels = [ax.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
labels = [label.replace("_", " ").capitalize() for label in labels]
lgd = fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.45, 1.22),
    fancybox=True,
    shadow=False,
    ncol=2,
    fontsize=legendsize,
    markerscale=2,
)

fig.savefig(
    "plot/Alfred/Alfred_subdataset_latency.png",
    bbox_extra_artists=(lgd,),
    bbox_inches="tight",
)
