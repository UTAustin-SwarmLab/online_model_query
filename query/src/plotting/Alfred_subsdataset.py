import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from swarm_visualizer.barplot import (
    plot_sns_grouped_barplot,  # noqa F401
)
from swarm_visualizer.utility.general_utils import (
    set_axis_infos,
    set_plot_properties,
)

sns.set()

xylabelsize = 28
legendsize = 20
ticksize = 24
bandits = {
    0: "FILM",
    1: "HiTUT",
    2: "HLSM",
}
colors = sns.color_palette("muted", 7)
# ["red", "blue", "green", "black", "tan", "orange", "purple"]
df = pd.read_csv("synced_data/csv/alfred_data/alfred_models_results.csv")
df.sort_values(by=["task_type"], inplace=True)

fig, ax = plt.subplots(figsize=(14, 10))
set_plot_properties()
# plot_sns_grouped_barplot(
#     df=df,
#     x_var="model",
#     y_var="SR",
#     hue="task_type",
#     # title_str=None,
#     ax=ax,
#     pal=colors,
#     # y_label="Accunracy of Models",
# )
sns.barplot(ax=ax, x="model", y="SR", palette=colors, hue="task_type", data=df)
ax.get_legend().remove()

plt.subplots_adjust(hspace=0.12)
set_axis_infos(
    ax,
    ylabel="Accuracy of Models",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    xlabel="Models",
    ticks_size=ticksize,
)

lines_labels = [ax.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
lgd = fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    fancybox=True,
    shadow=False,  # True,
    ncol=2,
    fontsize=legendsize,
    markerscale=2,
)

fig.savefig(
    "plot/Alfred/Alfred_subdataset.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
)
