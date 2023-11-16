import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils import scale_to_01_range

colors = ["red", "blue", "green", "black", "tan", "orange", "purple", "pink"]
markers = ["o", "v", "s", "p", "x", "D", "P", "*"]
data_path = "synced_data/csv/waymo/"
dataset_size = 13128
bandits = {
    0: "llava-v1.5-7b",
    1: "llava-v1.5-13b",
    2: "llava-v1.5-13b-lora",
}
emb_size = 768

### load embeddings
q_emb = np.load(data_path + "clip_emb_question.npy")  # 10x768
q_emb = np.tile(q_emb, (2000, 1))
img_emb = np.load(data_path + "clip_emb_img.npy")  # 2000x768
img_emb = np.repeat(img_emb, 10, axis=0)
arm_results = np.load(data_path + "arm_results.npy")
features = np.concatenate([q_emb, img_emb], axis=1)
# features = instruct_np

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
plt.savefig("./plot/Waymo/waymo_tsne_instruct_offload.png")
