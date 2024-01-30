import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from swarm_visualizer.utility.general_utils import (
    set_axis_infos,
)

df_path = "synced_data/csv/waymo/cloud-transmit-data-3.csv"
llava_results_path = "synced_data/csv/waymo/waymo_captions_llava-v1.5-13b.csv"
xylabelsize = 26
legendsize = 22
ticksize = 20
bins = 60

colors = sns.color_palette("muted", 4)

fig, ax_list = plt.subplot_mosaic(
    [["upload"], ["run time"], ["download"], ["total"]],
    figsize=(13, 15),
)
plt.subplots_adjust(hspace=0.35)

df = pd.read_csv(df_path)
df2 = pd.read_csv(llava_results_path)
### merge two dataframes
df = df2.join(df.set_index("waymo-index"))
df.dropna(inplace=True)
df["total"] = df["upload"] + df["download"] + df["inference time"]
df = df.iloc[5:, :]
print(df.head())
print(df.shape)

sns.histplot(data=df, x="upload", ax=ax_list["upload"], bins=bins)
set_axis_infos(
    ax_list["upload"],
    xlabel="Upload time (seconds)",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    ticks_size=ticksize,
    grid=True,
)
sns.histplot(data=df, x="inference time", ax=ax_list["run time"], bins=bins)
set_axis_infos(
    ax_list["run time"],
    xlabel="Model run time (seconds)",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    ticks_size=ticksize,
    grid=True,
)
sns.histplot(data=df, x="download", ax=ax_list["download"], bins=bins)
set_axis_infos(
    ax_list["download"],
    xlabel="Download time (seconds)",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    ticks_size=ticksize,
    grid=True,
)
sns.histplot(data=df, x="total", ax=ax_list["total"], bins=bins)
plt.grid()
set_axis_infos(
    ax_list["total"],
    xlabel="Total time (seconds)",
    xlabel_size=xylabelsize,
    ylabel_size=xylabelsize,
    ticks_size=ticksize,
)
fig.savefig("plot/Waymo/waymo_network_stat.pdf", bbox_inches="tight")
