import matplotlib.pyplot as plt
import numpy as np
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
    0: "vicuna-7b-v1.5",
    1: "falcon-180B",
    2: "falcon-180B-chat",
    3: "qCammel-70-x",
    4: "Llama-2-70b-instruct",
    5: "Llama-2-70b-instruct-v2",
    6: "StableBeluga-13B",
    7: "airoboros-l2-70b",
}

model_idx = [i for i in bandits.keys()]
colors = ["red", "blue", "green", "black", "tan", "orange", "purple", "pink"]
markers = ["o", "v", "s", "p", "x", "D", "P", "*"]


### load embeddings
question_np = np.load("synced_data/csv/mmlu/clip_emb_question.npy")
context_np = np.load("synced_data/csv/mmlu/clip_emb_choices.npy")
model_answer_np = np.load("synced_data/csv/mmlu/clip_emb_answer.npy")
X_complete = np.concatenate(
    (question_np, context_np, model_answer_np),
    axis=1,
)
print("X complete", X_complete.shape)

y_complete = np.load("synced_data/csv/mmlu/models_accnorm.npy")  # shape = [25256, 8]
print("y complete", y_complete.shape)
y_complete = y_complete[:, model_idx]


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
    label_text = bandits[idx]
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
    plt.savefig(f"./plot/mmlu_tsne_{bandits[idx]}.png")
