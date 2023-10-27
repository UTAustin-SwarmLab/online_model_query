from operator import itemgetter

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
    # 0: "deberta-v3-base-mrqa",
    1: "deberta-v3-base-squad2",
    2: "bigbird-base-trivia-itc",
    3: "distilbert-base-uncased-distilled-squad",
    4: "roberta-base-squad2-nq",
}

datasets = {
    "SQuAD": 0,
    "TriviaQA-web": 1,
    "NaturalQuestionsShort": 2,
    "NewsQA": 3,
    "SearchQA": 4,
    "HotpotQA": 5,
}

# batch size - algorithms will be refit after N rounds
batch_size = 100
dataset_size = 50000
percetile = 95
random_seed = 42
dataset = "mrqa"
max_iter = 2000
### set random seed
np.random.seed(random_seed)
colors = ["yellow", "red", "blue", "green", "black"]
markers = ["x", "o", "v", "s", "p"]

### idx
model_idx = [i for i in bandits.keys()]
dataset_idx = [i for i in datasets.values()]

### load embeddings
dataset_emb_list = []
squad_question_np = np.load("./synced_data/csv/mrqa/clip_emb_SQuAD_question.npy")
squad_context_np = np.load("./synced_data/csv/mrqa/clip_emb_SQuAD_context.npy")
squad_np = np.concatenate((squad_question_np, squad_context_np), axis=1)
trivia_question_np = np.load(
    "./synced_data/csv/mrqa/clip_emb_TriviaQA-web_question.npy"
)
trivia_context_np = np.load("./synced_data/csv/mrqa/clip_emb_TriviaQA-web_context.npy")
trivia_np = np.concatenate((trivia_question_np, trivia_context_np), axis=1)
natural_question_np = np.load(
    "./synced_data/csv/mrqa/clip_emb_NaturalQuestionsShort_question.npy"
)
natural_context_np = np.load(
    "./synced_data/csv/mrqa/clip_emb_NaturalQuestionsShort_context.npy"
)
natural_np = np.concatenate((natural_question_np, natural_context_np), axis=1)
news_question_np = np.load("./synced_data/csv/mrqa/clip_emb_NewsQA_question.npy")
news_context_np = np.load("./synced_data/csv/mrqa/clip_emb_NewsQA_context.npy")
news_np = np.concatenate((news_question_np, news_context_np), axis=1)
search_question_np = np.load("./synced_data/csv/mrqa/clip_emb_SearchQA_question.npy")
search_context_np = np.load("./synced_data/csv/mrqa/clip_emb_SearchQA_context.npy")
search_np = np.concatenate((search_question_np, search_context_np), axis=1)
hotpot_question_np = np.load("./synced_data/csv/mrqa/clip_emb_HotpotQA_question.npy")
hotpot_context_np = np.load("./synced_data/csv/mrqa/clip_emb_HotpotQA_context.npy")
hotpot_np = np.concatenate((hotpot_question_np, hotpot_context_np), axis=1)

dataset_emb_list.append(squad_np)
dataset_emb_list.append(trivia_np)
dataset_emb_list.append(natural_np)
dataset_emb_list.append(news_np)
dataset_emb_list.append(search_np)
dataset_emb_list.append(hotpot_np)
dataset_emb_list = itemgetter(*dataset_idx)(dataset_emb_list)

X_complete = np.concatenate(
    (dataset_emb_list),
    axis=0,
)
arr = np.random.choice(np.arange(X_complete.shape[0]), dataset_size, replace=True)
print("X complete", X_complete.shape)
X = X_complete[arr, :]

dataset_y_list = []
squad_exact_np = np.load("./synced_data/csv/mrqa/SQuAD_exact.npy")
trivia_exact_np = np.load("./synced_data/csv/mrqa/TriviaQA-web_exact.npy")
natural_exact_np = np.load("./synced_data/csv/mrqa/NaturalQuestionsShort_exact.npy")
news_exact_np = np.load("./synced_data/csv/mrqa/NewsQA_exact.npy")
search_exact_np = np.load("./synced_data/csv/mrqa/SearchQA_exact.npy")
hotpot_exact_np = np.load("./synced_data/csv/mrqa/HotpotQA_exact.npy")
dataset_y_list.append(squad_exact_np)
dataset_y_list.append(trivia_exact_np)
dataset_y_list.append(natural_exact_np)
dataset_y_list.append(news_exact_np)
dataset_y_list.append(search_exact_np)
dataset_y_list.append(hotpot_exact_np)
dataset_y_list = itemgetter(*dataset_idx)(dataset_y_list)

y_complete = (
    np.concatenate(
        (dataset_y_list),
        axis=0,
    )
    / 100
)
y_complete = y_complete
print("y complete", y_complete.shape)
is_1 = np.where(y_complete == 1)

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
    plt.savefig(f"./plot/QA_tsne_{bandits[idx].split('-')[0]}.png")
