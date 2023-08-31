### use clip to get emneddings of the imagenet dataset
### use the embeddings to plot a t-SNE plot
# from transformers import CLIPProcessor, CLIPModel
import clip
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image
from sklearn.manifold import TSNE
from transformers import ViTForImageClassification

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
modulo = 100

### load model and dataset
model, preprocess = clip.load("ViT-B/32", device=device)
model_hf = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(
    device
)

dataset = load_dataset(
    "imagenet-1k", split="validation", streaming=False, use_auth_token=True
).with_format("numpy")
# dataset = dataset.shard(num_shards=10, index=0)
dataset = dataset.filter(lambda example: example["label"] % modulo == 0)
labels = []
for id in range(0, 1000, modulo):
    labels.append((id, model_hf.config.id2label[id].split(",")[0]))
print("labels: ", labels)
print("num of labels: ", len(labels), "num of data: ", len(dataset))

### randomly select 10 colors
import random

random.seed(0)
colors = []
for i in range(10):
    colors.append("#%06X" % random.randint(0, 0xFFFFFF))

features, data_label = [], []
for i in range(len(dataset)):
    np_image, label = dataset[i]["image"], dataset[i]["label"].item()
    if len(np_image.shape) == 2:
        np_image = np.stack((np_image,) * 3, axis=-1)
    image = preprocess(Image.fromarray(np_image, "RGB")).unsqueeze(0).to(device)
    # print(i, image.shape, label)
    image_features = model.encode_image(image)
    image_features = image_features.reshape((-1, 1)).detach().numpy()
    features.append(image_features)
    data_label.append(label)

features = np.concatenate(features, axis=1).T
tsne = TSNE(n_components=2).fit_transform(features)


### scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = np.max(x) - np.min(x)
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


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
for i, (label_id, label_text) in enumerate(labels):
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(data_label) if l == label_id]
    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # convert the class color to matplotlib format
    color = colors[i]
    print(indices, label_text, color)

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=[color] * len(indices), label=label_text)

# build a legend using the labels we set previously
ax.legend(fontsize="8", loc="upper left")

# finally, show the plot
plt.savefig("./plot/imagenet_tsne.png")
