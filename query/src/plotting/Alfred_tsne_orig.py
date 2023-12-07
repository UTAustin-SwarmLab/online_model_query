import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from utils import scale_to_01_range

colors = ["red", "blue", "green", "black", "tan", "orange", "purple", "pink"]
markers = ["o", "v", "s", "p", "x", "D", "P", "*"]

### load dataframes
task_df = (
    pd.read_csv("synced_data/csv/alfred_data/alfred_merged_valid_language_goal.csv")
    .set_index(["task_idx", "repeat_idx"])
    .sort_index()
)
task_cate_df = (
    pd.read_csv("synced_data/csv/alfred_data/alfred_models_results.csv")
    .drop_duplicates(subset=["task_idx", "repeat_idx"], keep="first")
    .set_index(["task_idx", "repeat_idx"])
    .sort_index()
)

### join task_df and task_cate_df
task_df = task_df.merge(
    task_cate_df,
    left_on=["task_idx", "repeat_idx"],
    how="inner",
    right_on=["task_idx", "repeat_idx"],
    validate="one_to_one",
    suffixes=("", "_y"),
).drop(columns=["split_y"])

### load embeddings
task_dict = pickle.load(open("synced_data/csv/alfred_data/clip_emb_instruct.pkl", "rb"))
task_dict = {key: task_dict[key] for key in task_df.index}
task_np = np.array(list(task_dict.values()))
print("Task dataframe: ", task_df.shape)

floorplan_dict = pickle.load(open("synced_data/csv/alfred_data/floor_plan.pkl", "rb"))

### t-sne floorplan
features = floorplan_dict.values()
features = np.concatenate([f for f in features], axis=0)
print("features", features.shape)
tsne = TSNE(n_components=2).fit_transform(features)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

### plot the result
# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

# convert the class color to matplotlib format
color = "blue"
label_text = "Floor Plans"
marker = "o"

# add a scatter plot with the corresponding color and label
ax.scatter(
    tx,
    ty,
    c=color,  # * len(indice),
    label=label_text,
    s=12,
    marker=marker,
)

# build a legend using the labels we set previously
ax.legend(fontsize="8", loc="upper left")

# finally, show the plot
plt.savefig("./plot/alfred_tsne_floorplan.png")


### t-sne task instructions
features = task_np
print("features", features.shape)
tsne = TSNE(n_components=2).fit_transform(features)
tx = tsne[:, 0]
ty = tsne[:, 1]
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

task_types = task_df["task_type"].unique().tolist()
fig = plt.figure()
ax = fig.add_subplot(111)

for ii, task_type in enumerate(task_types):
    task_batch_indices = [
        row["task_type"] == task_type for _, row in task_df.iterrows()
    ]
    print(ii, task_type, sum(task_batch_indices))

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    x = tx[task_batch_indices]
    y = ty[task_batch_indices]

    # convert the class color to matplotlib format
    color = colors[ii]
    label_text = task_type
    marker = markers[ii]

    # add a scatter plot with the corresponding color and label
    ax.scatter(
        x,
        y,
        c=color,  # * len(indice),
        label=label_text,
        s=5,
        marker=marker,
    )

# build a legend using the labels we set previously
ax.legend(fontsize="8", loc="upper left")

# finally, show the plot
plt.savefig("./plot/Alfred/alfred_tsne_instruct.png")
