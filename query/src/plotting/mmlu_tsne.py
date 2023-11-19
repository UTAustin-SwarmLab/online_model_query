# plot tsne for mmlu in terms of sub datasets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from utils import scale_to_01_range

# sns.set()
titlesize = 24
xyticksize = 20
legendsize = 24
markersize = 55
subset_map = {
    # "0": "arc:challenge",
    # "1": "hellaswag",
    # "2": "hendrycksTest-abstract_algebra",
    # "3": "hendrycksTest-anatomy",
    # "4": "hendrycksTest-astronomy",
    # "5": "hendrycksTest-business_ethics",
    # "6": "hendrycksTest-clinical_knowledge",
    # "7": "hendrycksTest-college_biology",
    # "8": "hendrycksTest-college_chemistry",
    # "9": "hendrycksTest-college_computer_science",
    # "10": "hendrycksTest-college_mathematics",
    # "11": "hendrycksTest-college_medicine",
    # "12": "hendrycksTest-college_physics",
    # "13": "hendrycksTest-computer_security",
    # "14": "hendrycksTest-conceptual_physics",
    # "15": "hendrycksTest-econometrics",
    # "16": "hendrycksTest-electrical_engineering",
    "17": "hendrycksTest-elementary_mathematics",
    # "18": "hendrycksTest-formal_logic",
    # "19": "hendrycksTest-global_facts",
    # "20": "hendrycksTest-high_school_biology",
    # "21": "hendrycksTest-high_school_chemistry",
    # "22": "hendrycksTest-high_school_computer_science",
    # "23": "hendrycksTest-high_school_european_history",
    # "24": "hendrycksTest-high_school_geography",
    # "25": "hendrycksTest-high_school_government_and_politics",
    "26": "hendrycksTest-high_school_macroeconomics",
    # "27": "hendrycksTest-high_school_mathematics",
    # "28": "hendrycksTest-high_school_microeconomics",
    # "29": "hendrycksTest-high_school_physics",
    "30": "hendrycksTest-high_school_psychology",
    # "31": "hendrycksTest-high_school_statistics",
    # "32": "hendrycksTest-high_school_us_history",
    # "33": "hendrycksTest-high_school_world_history",
    # "34": "hendrycksTest-human_aging",
    # "35": "hendrycksTest-human_sexuality",
    # "36": "hendrycksTest-international_law",
    # "37": "hendrycksTest-jurisprudence",
    # "38": "hendrycksTest-logical_fallacies",
    # "39": "hendrycksTest-machine_learning",
    # "40": "hendrycksTest-management",
    # "41": "hendrycksTest-marketing",
    # "42": "hendrycksTest-medical_genetics",
    # "43": "hendrycksTest-miscellaneous",
    # "44": "hendrycksTest-moral_disputes",
    "45": "hendrycksTest-moral_scenarios",
    # "46": "hendrycksTest-nutrition",
    # "47": "hendrycksTest-philosophy",
    # "48": "hendrycksTest-prehistory",
    # "49": "hendrycksTest-professional_accounting",
    "50": "hendrycksTest-professional_law",
    # "51": "hendrycksTest-professional_medicine",
    "52": "hendrycksTest-professional_psychology",
    # "53": "hendrycksTest-public_relations",
    # "54": "hendrycksTest-security_studies",
    # "55": "hendrycksTest-sociology",
    # "56": "hendrycksTest-us_foreign_policy",
    # "57": "hendrycksTest-virology",
    # "58": "hendrycksTest-world_religions",
}

data_path = "synced_data/csv/mmlu/"

# batch size - algorithms will be refit after N rounds
dataset = "mmlu"


def plot_mmlu_tsne(save=False, ax_=None):
    ### load subsets ###
    data = pd.read_csv(data_path + "vicuna-7b-v1.5_nochoice.csv")
    selected_indices = []
    idx = 0
    filtered_indices = 0
    subset2idx = {}
    for subset in subset_map.values():
        subset2idx[subset] = []
    for _, row in data.iterrows():
        if row["subdataset"] in subset_map.values():
            selected_indices.append(idx)
            subset2idx[row["subdataset"]].append(filtered_indices)
            filtered_indices += 1
        idx += 1

    subset_cnt = [(k, len(v)) for k, v in subset2idx.items()]
    subset_cnt.sort(key=lambda x: x[1], reverse=True)
    print("size of each subset: ", subset_cnt, "\n")

    ### load embeddings
    question_np = np.load(data_path + "clip_emb_question.npy")[selected_indices, :]
    context_np = np.load(data_path + "clip_emb_choices.npy")[selected_indices, :]
    # model_answer_np = (np.load(data_path + "clip_emb_answer.npy"))[selected_indices, :]

    X_complete = np.concatenate(
        (question_np, context_np),
        axis=1,
    )
    features = X_complete
    tsne = TSNE(n_components=2).fit_transform(features)

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    ### plot the result
    # initialize a matplotlib plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    # NUM_COLORS = len(subset_map.values())
    # cm = plt.get_cmap("gist_rainbow")
    # colors = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    # colors = sns.color_palette("hls", 6)
    colors = ["red", "blue", "green", "black", "orange", "purple", "pink"]

    # for every subset, we plot the points separately
    for idx, subset in enumerate(subset_map.values()):
        # find the samples of the current class in the data
        indice = subset2idx[subset]
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indice)
        current_ty = np.take(ty, indice)

        # convert the class color to matplotlib format
        color = colors[idx]
        label_text = subset.replace("hendrycksTest-", "")
        marker = "o"

        # add a scatter plot with the corresponding color and label
        ax.scatter(
            current_tx,
            current_ty,
            c=color,  # * len(indice),
            label=label_text,
            s=markersize,
            marker=marker,
        )
    if save:
        lines_labels = [ax.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        lgd = ax.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            fancybox=True,
            shadow=True,
            ncol=2,
            fontsize=legendsize,
            markerscale=1.5,
        )
        # ax.set_title("T-SNE of MMLU embeddings", fontsize=titlesize)
        plt.xticks(fontsize=xyticksize)
        plt.yticks(fontsize=xyticksize)

        plt.savefig(
            "./plot/mmlu/mmlu_tsne.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
        )
        return
    else:
        plt.show()
        return ax


if __name__ == "__main__":
    plot_mmlu_tsne(save=True)
