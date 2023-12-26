import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    4: "LLaMa-2-70B",
    2: "Falcon-180B",
}
xylabelsize = 26
legendsize = 22
ticksize = 20
markersize = 250

sns.set()
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)

time = []
for key, value in bandits.items():
    latency = pd.read_csv(f"synced_data/csv/mmlu/{value}_answer_time.csv")
    latency = np.array(latency["answer_time"] + latency["load_time"])
    time.append(np.mean(latency))
    print(f"{value}: {np.mean(latency)}")

acc = []
for key, value in bandits.items():
    accuracy = pd.read_csv(f"synced_data/csv/mmlu/{value}_nochoice.csv")
    accuracy = np.array(accuracy["acc_norm"])
    acc.append(np.mean(accuracy))
    print(f"{value}: {np.mean(accuracy)}")

for i in zip(bandits.values(), time, acc, names.values()):
    plt.scatter(i[1], i[2], label=i[3], s=markersize)

plt.legend(fontsize=legendsize)
set_axis_infos(
    ax=ax,
    xlabel="Run Time (seconds)",
    ylabel="Accuracy",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    ticks_size=ticksize,
    yticks=[0.6, 0.65, 0.7, 0.75, 0.8],
)

plt.savefig("plot/mmlu/mmlu_time_vs_acc.pdf", bbox_inches="tight")
