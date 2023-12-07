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
colors = sns.color_palette("muted", 4)
xylabelsize = 28
legendsize = 20
ticksize = 24
bandits = {
    0: "vicuna-7b-v1.5",
    # 1: "falcon-180B",
    2: "falcon-180B-chat",
    # 3: "qCammel-70-x",
    4: "Llama-2-70b-instruct",
    # 5: "Llama-2-70b-instruct-v2",
    6: "StableBeluga-13B",
    # 7: "airoboros-l2-70b",
}
models_labels = ["Vicuna-7B", "Falcon-180B", "LLaMA-2-70B", "StableBeluga-13B"]
subtasks = {
    "hendrycksTest-high_school_chemistry",
    "hendrycksTest-high_school_computer_science",
    "hendrycksTest-high_school_macroeconomics",
    "hendrycksTest-high_school_government_and_politics",
}

dfs = {}
for model in bandits.values():
    df = pd.read_csv(f"synced_data/csv/mmlu/{model}_nochoice.csv")
    # filter df by subdataset
    idx = df["subdataset"].isin(subtasks)
    df = df[idx]
    dfs[model] = df

df = pd.concat(dfs.values(), axis=0)

fig, ax = plt.subplots(figsize=(14, 10))
set_plot_properties()
# plot_sns_grouped_barplot(
#     df=df,
#     x_var="model",
#     y_var="acc_norm",
#     hue="subdataset",
#     # title_str=None,
#     ax=ax,
#     # y_label="Accuracy of Models",
# )
sns.barplot(ax=ax, x="model", y="acc_norm", palette=colors, hue="subdataset", data=df)
ax.get_legend().remove()
ax.set_xticklabels(models_labels)

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
labels = [label.replace("hendrycksTest-", "") for label in labels]
lgd = fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    fancybox=True,
    shadow=True,
    ncol=2,
    fontsize=legendsize,
    markerscale=2,
)

fig.savefig(
    "plot/mmlu/mmlu_subdataset.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
)
