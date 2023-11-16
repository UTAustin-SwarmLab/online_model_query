import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils import scale_to_01_range

colors = ["red", "blue", "green", "black", "tan", "orange", "purple", "pink"]
markers = ["o", "v", "s", "p", "x", "D", "P", "*"]
data_path = "synced_data/csv/alfred_data/"
dataset_size = 13128
bandits = {
    0: "FILM",
    1: "HiTUT",
    2: "HLSM",
}
alpha = 0.05
beta = 0.005
emb_size = 768

### load embeddings
emb = np.load(data_path + "clip_emb.npy")
instruct_np = emb[:, :emb_size]
ll_instruct_np = emb[:, emb_size : emb_size * 4]
floorplan_np = emb[:, emb_size * 4 :]
# features = np.concatenate([instruct_np, floorplan_np], axis=1)
features = instruct_np
arm_results = np.load(data_path + "arm_results.npy")

### load token length
token_len = np.load(data_path + "instruct_token_length.npy")  # (13128,)
token_len = token_len.reshape(dataset_size, 1)  # (13128, 1)
token_len = np.repeat(token_len, len(bandits), axis=1)  # (39384, 3)
token_len[:, 0] = 0

gc = arm_results[1, :, :]
L_ratio = arm_results[3, :, :] / np.maximum(arm_results[2, :, :], arm_results[3, :, :])
L = arm_results[2, :, :]

# self.arm_results = 0.5 * gc + 0.5 * gc * L_ratio - beta * token_len
arm_results = 0.5 * gc - alpha * np.log10(L) - beta * token_len
select_arm = np.argmax(arm_results, axis=1)
select_idx = select_arm == 0
print(select_arm.shape, select_arm[:20])
print(select_idx.shape, select_idx[:20])

### t-sne task instructions
print("features", features.shape)
tsne = TSNE(n_components=2).fit_transform(features)
tx = tsne[:, 0]
ty = tsne[:, 1]
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

fig = plt.figure()
ax = fig.add_subplot(111)

### plot data points which should be offloaded to local
# extract x and y coordinates representing the positions of the images on T-SNE plot
x = tx[select_idx]
y = ty[select_idx]

# add a scatter plot with the corresponding color and label
ax.scatter(
    x,
    y,
    c=colors[0],
    label="local model",
    s=5,
    marker=markers[0],
)

### plot data points which should be offloaded to cloud
# extract x and y coordinates representing the positions of the images on T-SNE plot
x = tx[~select_idx]
y = ty[~select_idx]

# add a scatter plot with the corresponding color and label
ax.scatter(
    x,
    y,
    c=colors[1],
    label="cloud model",
    s=5,
    marker=markers[1],
)

# build a legend using the labels we set previously
ax.legend(fontsize="8", loc="upper left")

# finally, show the plot
plt.savefig("./plot/Alfred/alfred_tsne_instruct_offload.png")
