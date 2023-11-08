import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


### scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = np.max(x) - np.min(x)
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


bandits = {
    0: "FILM",
    1: "HiTUT",
    2: "HLSM",
}

data_path = "synced_data/csv/alfred_data/"
model_path = "synced_data/cumulative_reward/"
colors = ["red", "blue", "green", "black", "tan", "orange", "purple", "pink"]
markers = ["o", "v", "s", "p", "x", "D", "P", "*"]
reward_metric = "SR"

# batch size - algorithms will be refit after N rounds
batch_size = 100
dataset_size = 50000
percetile = 95
random_seed = 42
dataset = "mrqa"
max_iter = 2000

### load data
instruction_dict = pickle.load(open(data_path + "clip_emb_instruct.pkl", "rb"))
alfred_data = pd.read_csv(data_path + "alfred_valid_language_goal.csv")
arm_results_df = pd.read_csv(data_path + "alfred_models_results.csv")

emb = []
y = np.zeros((4, len(alfred_data), len(bandits)), dtype=np.float32)
for _, row in alfred_data.iterrows():
    task_id = row["task_idx"]
    repeat_idx = row["repeat_idx"]
    floorplan = row["task_floor"]
    emb.append(instruction_dict[(task_id, repeat_idx)])
    for i, model in bandits.items():
        result_row = arm_results_df.loc[
            (arm_results_df["task_idx"] == task_id)
            & (arm_results_df["repeat_idx"] == repeat_idx % 10)
            & (arm_results_df["model"] == model)
        ]
        sr = result_row["SR"].iloc[0]
        gc = result_row["GC"].iloc[0]
        L = result_row["L"].iloc[0]
        L_demo = result_row["L*"].iloc[0]
        y[0, _, i] = sr
        y[1, _, i] = gc
        y[2, _, i] = L
        y[3, _, i] = L_demo

X_complete = np.array(emb)  ### only the orig instructions
arm_results = y

### load embeddings (merged instructions)
# X_complete = np.load(data_path + "clip_emb.npy")
# arm_results = np.load(data_path + "arm_results.npy")
if reward_metric == "GC":
    y_complete = arm_results[1, :, :]
elif reward_metric == "PLWGC":
    y_complete = arm_results[1, :, :] * (
        arm_results[3, :, :] / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
    )
elif reward_metric == "SR":
    y_complete = arm_results[0, :, :]
elif reward_metric == "PLWSR":
    y_complete = arm_results[0, :, :] * (
        arm_results[3, :, :] / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
    )
elif reward_metric == "gc and plw":
    gc = arm_results[1, :, :]
    plw = arm_results[1, :, :] * (
        arm_results[3, :, :] / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
    )
    y_complete = 0.5 * gc + 0.5 * plw

### delete low level instructions and floorplan
X_complete = X_complete[:, :768]
print("reward_metric: ", reward_metric)
print("X_complete shape: ", X_complete.shape)
print("y_complete shape: ", y_complete.shape)
input("Press Enter to continue...")
is_1 = np.where(y_complete == 1)
is_0 = np.where(y_complete == 0)

features = X_complete
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

# for every class, we'll add a scatter plot separately
for idx, value in bandits.items():
    # find the samples of the current class in the data
    indice = np.where(y_complete[:, idx] == 1)[0]
    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indice)
    current_ty = np.take(ty, indice)

    # convert the class color to matplotlib format
    color = colors[idx]
    label_text = bandits[idx].split("-")[0]
    marker = markers[idx]

    # add a scatter plot with the corresponding color and label
    ax.scatter(
        current_tx,
        current_ty,
        c=color,  # * len(indice),
        label=label_text,
        s=5,
        marker=marker,
    )

    # build a legend using the labels we set previously
    ax.legend(fontsize="8", loc="upper left")

    # finally, show the plot
    plt.savefig(f"./plot/Alfred_tsne_{bandits[idx]}.png")

# plot failed cases
current_tx = np.take(tx, is_0)
current_ty = np.take(ty, is_0)
color = colors[idx + 1]
label_text = "failed"
marker = markers[idx + 1]

ax.scatter(
    current_tx,
    current_ty,
    c=color,  # * len(indice),
    label=label_text,
    s=5,
    marker=marker,
)
ax.legend(fontsize="8", loc="upper left")
plt.savefig("./plot/Alfred_tsne_failed.png")

plt.savefig("./plot/Alfred_tsne_bandits.png")
