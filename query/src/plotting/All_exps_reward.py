import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Alfred_plot import plot_alfred
from mmlu_latency_plot import plot_mmlu_lat
from mmlu_plot import plot_mmlu
from swarm_visualizer.utility.general_utils import (
    set_axis_infos,
)
from Waymo_latency_plot import plot_waymo_lat
from Waymo_plot import plot_waymo

sns.set()
xylabelsize = 20
titlesize = 26
legendsize = 20
ticksize = 18

fig, ax_list = plt.subplot_mosaic(
    [["mmlu_sr", "waymo_sr", "alfred_sr"], ["mmlu_lat", "waymo_lat", "alfred_lat"]],
    figsize=(20, 10),
)
plt.subplots_adjust(hspace=0.12)
ax_mmlu_sr = plot_mmlu(ax_=ax_list["mmlu_sr"])
ax_waymo_sr = plot_waymo(ax_=ax_list["waymo_sr"])
ax_alfred_sr = plot_alfred(ax_=ax_list["alfred_sr"])
ax_mmlu_lat = plot_mmlu_lat(ax_=ax_list["mmlu_lat"])
ax_waymo_lat = plot_waymo_lat(ax_=ax_list["waymo_lat"])
ax_alfred_lat = plot_alfred(ax_=ax_list["alfred_lat"], reward_metric="GC+PLW")


set_axis_infos(
    ax_mmlu_sr,
    ylabel="Cumulative Mean\nSuccess Rate",
    title_str=r"$\bf{MMLU}$",
    ylim=(0.59, 0.91),
    xticks=list(np.arange(0, 12501, 4000)),
    yticks=list(np.arange(0.6, 0.91, 0.1)),
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    title_size=titlesize,
    ticks_size=ticksize,
)
set_axis_infos(
    ax_waymo_sr,
    title_str=r"$\bf{Waymo}$",
    ylim=(0.74, 0.96),
    xticks=list(np.arange(0, 20001, 5000)),
    yticks=list(np.arange(0.75, 0.96, 0.1)),
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    title_size=titlesize,
    ticks_size=ticksize,
)
set_axis_infos(
    ax_alfred_sr,
    title_str=r"$\bf{ALFRED}$",
    ylim=(0.19, 0.53),
    xticks=list(np.arange(0, 13000, 4000)),
    yticks=list(np.arange(0.2, 0.51, 0.1)),
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    title_size=titlesize,
    ticks_size=ticksize,
)
set_axis_infos(
    ax_mmlu_lat,
    xlabel="Number of Data Observed",
    ylabel="Cumulative Mean Reward\n with Latency and Costs",
    ylim=(0.49, 0.81),
    xticks=list(np.arange(0, 12501, 4000)),
    yticks=list(np.arange(0.5, 0.81, 0.1)),
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    title_size=titlesize,
    ticks_size=ticksize,
)
set_axis_infos(
    ax_waymo_lat,
    xlabel="Number of Data Observed",
    ylim=(0.64, 0.86),
    xticks=list(np.arange(0, 20001, 5000)),
    yticks=list(np.arange(0.6, 0.86, 0.1)),
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    title_size=titlesize,
    ticks_size=ticksize,
)
set_axis_infos(
    ax_alfred_lat,
    xlabel="Number of Data Observed",
    ylim=(0.0, 0.22),
    xticks=list(np.arange(0, 13000, 4000)),
    yticks=list(np.arange(0.0, 0.21, 0.1)),
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    title_size=titlesize,
    ticks_size=ticksize,
)

lines_labels = [ax_mmlu_sr.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
lgd = fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    fancybox=True,
    shadow=True,
    ncol=5,
    fontsize=legendsize,
    markerscale=2,
)

fig.savefig("plot/results.pdf", bbox_extra_artists=(lgd,), bbox_inches="tight")
