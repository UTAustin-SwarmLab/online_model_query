import matplotlib.pyplot as plt
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
bandits = {
    0: "FILM",
    1: "HiTUT",
    2: "HLSM",
}
# ["red", "blue", "green", "black", "tan", "orange", "purple"]
df = pd.read_csv("synced_data/csv/alfred_data/alfred_models_results.csv")
df.sort_values(by=["task_type"], inplace=True)

fig, ax = plt.subplots(figsize=(18, 8))
plt.grid()
sns.barplot(
    ax=ax,
    x="model",
    y="SR",
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
    ylabel="Accuracy of models",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    xlabel="Models",
    ticks_size=ticksize,
)

lines_labels = [ax.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
labels = [label.replace("_", " ").capitalize() for label in labels]
lgd = fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.47, 1.22),
    fancybox=True,
    shadow=False,  # True,
    ncol=2,
    fontsize=legendsize,
    markerscale=2,
)

fig.savefig(
    "plot/Alfred/Alfred_subdataset.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
)
