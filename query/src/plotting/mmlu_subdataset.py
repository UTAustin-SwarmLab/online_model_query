import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from swarm_visualizer.barplot import (
    plot_sns_grouped_barplot,  # noqa F401
)
from swarm_visualizer.utility.general_utils import (
    set_axis_infos,
)

colors = sns.color_palette("tab10")
xylabelsize = 32
legendsize = 28
ticksize = 32
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

fig, ax = plt.subplots(figsize=(18, 8))
plt.grid()
sns.barplot(
    ax=ax,
    x="model",
    y="acc_norm",
    palette=colors,
    hue="subdataset",
    data=df,
    hue_order=[
        "hendrycksTest-high_school_government_and_politics",
        "hendrycksTest-high_school_computer_science",
        "hendrycksTest-high_school_macroeconomics",
        "hendrycksTest-high_school_chemistry",
    ],
)
ax.get_legend().remove()
models_labels = ["Vicuna-7B", "StableBeluga-13B", "   LLaMA-2-70B", "Falcon-180B"]
ax.set_xticklabels(models_labels)
ax.set_ylim(0.0, 0.99)

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
labels = [
    label.replace("hendrycksTest-", "").replace("_", " ").capitalize()
    for label in labels
]
lgd = fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.45, 1.07),
    ncol=2,
    fontsize=legendsize,
    markerscale=2,
    facecolor=(1, 1, 1, 0.1),
)
fig.savefig(
    "plot/mmlu/mmlu_subdataset.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
)
