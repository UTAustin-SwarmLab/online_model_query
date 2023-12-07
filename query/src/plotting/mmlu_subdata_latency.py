import matplotlib.pyplot as plt
import numpy as np
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
alpha = 0.03
beta = 0.003


all_data = pd.read_csv("synced_data/csv/mmlu/vicuna-7b-v1.5_nochoice.csv")
selected_indices = []
idx = 0
for _, row in all_data.iterrows():
    if row["subdataset"] in subtasks:
        selected_indices.append(idx)
    idx += 1
model_idx = [i for i in bandits.keys()]
arm_results = np.load("synced_data/csv/mmlu/models_accnorm.npy")
arm_results = arm_results[:, model_idx]

model_latency = np.zeros_like(arm_results)
idx = 0
for key, value in bandits.items():
    latency = pd.read_csv(f"synced_data/csv/mmlu/{value}_answer_time.csv")
    latency = np.array(latency["answer_time"] + latency["load_time"])
    repeat_cnt = arm_results.shape[0] // len(latency) + 1
    latency = np.tile(latency, repeat_cnt)
    model_latency[:, idx] = latency[: arm_results.shape[0]]
    idx += 1

token_len = np.zeros_like(arm_results)
token_length = np.load("synced_data/csv/mmlu/question_token_length.npy")
token_len[:, 1:] = token_length[:, np.newaxis]

dfs = {}
arm_results = arm_results[selected_indices, :]
model_latency = model_latency[selected_indices, :]
token_len = token_len[selected_indices, :]
for ii, model in enumerate(bandits.values()):
    df = pd.read_csv(f"synced_data/csv/mmlu/{model}_nochoice.csv")
    # filter df by subdataset
    idx = df["subdataset"].isin(subtasks)
    df = df[idx]
    dfs[model] = df
    reward = (
        arm_results[:, ii]
        - alpha * np.log10(model_latency[:, ii])
        - beta * token_len[:, ii]
    )
    df["reward_latency"] = reward
df = pd.concat(dfs.values(), axis=0)

fig, ax = plt.subplots(figsize=(14, 10))
set_plot_properties()
sns.barplot(
    ax=ax, x="model", y="reward_latency", palette=colors, hue="subdataset", data=df
)
ax.get_legend().remove()
ax.set_xticklabels(models_labels)

plt.subplots_adjust(hspace=0.12)
set_axis_infos(
    ax,
    ylabel="Mean Rewards of Models\nwith Latency and Costs",
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
    "plot/mmlu/mmlu_subdataset_latency.png",
    bbox_extra_artists=(lgd,),
    bbox_inches="tight",
)