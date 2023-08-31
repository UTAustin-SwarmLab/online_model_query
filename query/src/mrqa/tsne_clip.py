import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

datasets = ["SQuAD", "TriviaQA-web", "NaturalQuestionsShort", "NewsQA"]  #
dataset_size_list = (10507, 7785, 12836, 4212)
dataset_size_list = np.cumsum(dataset_size_list)
print(dataset_size_list)
colors = ["red", "blue", "green", "black"]
markers = ["o", "v", "s", "p"]

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

features = np.concatenate((squad_np, trivia_np, natural_np, news_np), axis=0)
print(features.shape)
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
for i in range(len(datasets)):
    # find the samples of the current class in the data
    if i == 0:
        indices = [j for j in range(dataset_size_list[i])]
        print(dataset_size_list[i])
    else:
        indices = [j for j in range(dataset_size_list[i - 1], dataset_size_list[i])]
        print(dataset_size_list[i - 1], dataset_size_list[i])
    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # convert the class color to matplotlib format
    color = colors[i]
    label_text = datasets[i]
    marker = markers[i]

    # add a scatter plot with the corresponding color and label
    ax.scatter(
        current_tx,
        current_ty,
        c=color,  # * len(indices),
        label=label_text,
        s=5,
        marker=marker,
    )

# build a legend using the labels we set previously
ax.legend(fontsize="8", loc="upper left")

# finally, show the plot
plt.savefig("./plot/QA_tsne.png")
