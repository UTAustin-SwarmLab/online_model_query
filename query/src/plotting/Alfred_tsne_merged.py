import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from utils import scale_to_01_range

xyticksize = 32
legendsize = 28
markersize = 50


def plot_alfred_tsne():
    # colors = ["red", "blue", "green", "black", "tan", "orange", "purple", "pink"]
    colors = sns.color_palette("tab10")
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
    for i, row in task_df.iterrows():
        task_df.loc[i, "task_type"] = task_cate_df.loc[(i[0], i[1] % 10), "task_type"]

    print(task_df.head(), task_df.shape, task_df.columns)

    ### load embeddings
    task_dict = pickle.load(
        open("synced_data/csv/alfred_data/clip_emb_instruct.pkl", "rb")
    )
    task_dict = {key: task_dict[key] for key in task_df.index}
    task_np = np.array(list(task_dict.values()))
    print("Task dataframe: ", task_df.shape)

    ### t-sne task instructions
    features = task_np
    print("features", features.shape)
    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    task_types = task_df["task_type"].unique().tolist()
    task_types.sort()

    # Define the order in which you want the legends to appear
    legend_order = [
        "pick_clean_then_place_in_recep",
        "look_at_obj_in_light",
        "pick_and_place_simple",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
        "pick_and_place_with_movable_recep",
    ]

    # Ensure that task_types is sorted according to legend_order
    task_types = sorted(task_types, key=lambda x: legend_order.index(x))

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)

    for ii, task_type in enumerate(task_types):
        task_batch_indices = [
            row["task_type"] == task_type for _, row in task_df.iterrows()
        ]
        print(ii, task_type, sum(task_batch_indices))

        # extract x and y coordinates representing the positions of the images on T-SNE plot
        x = tx[task_batch_indices]
        y = ty[task_batch_indices]

        ### plot the result
        # initialize a matplotlib plot

        # convert the class color to matplotlib format
        color = colors[ii]
        label_text = task_type.replace("_", " ").capitalize()
        marker = markers[0]

        # add a scatter plot with the corresponding color and label
        ax.scatter(
            x,
            y,
            c=color,  # * len(indice),
            label=label_text,
            s=markersize,
            marker=marker,
        )

    # sort both labels and lines by labels in legend_order's order
    legend_order = [
        "pick_clean_then_place_in_recep",
        "look_at_obj_in_light",
        "pick_and_place_simple",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
        "pick_and_place_with_movable_recep",
    ]
    zipped = zip(legend_order, colors, markers)
    zipped = sorted(zipped, key=lambda x: x[0])
    legend_order, colors, markers = zip(*zipped)
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    print(legend_order)
    print(labels)
    print(lines)
    lgd = ax.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.37),
        fancybox=True,
        shadow=True,
        ncol=2,
        fontsize=legendsize,
        markerscale=2.5,
    )
    ax.set_xlabel("Normalized projected dimension 1", fontsize=xyticksize)
    ax.set_ylabel("Normalized projected dimension 2", fontsize=xyticksize)
    plt.xticks(fontsize=xyticksize)
    plt.yticks(fontsize=xyticksize)

    # finally, show the plot
    plt.savefig(
        "./plot/Alfred/alfred_tsne_instruct_merged.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    return


if __name__ == "__main__":
    plot_alfred_tsne()
