import numpy as np
import pandas as pd

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

### load and then train model if it exists
model_subsets = pd.read_csv("synced_data/csv/mmlu/model_subsets.csv")
subsets = model_subsets["subdataset"].unique()

### sort models by the index in the bandits dict
models = bandits.values()
models = sorted(models, key=lambda x: list(bandits.values()).index(x))

experts = []

acc_matrix = np.zeros((len(subsets), len(models)))
for i, subset in enumerate(subsets):
    for j, model in enumerate(models):
        row = model_subsets[
            (model_subsets["subdataset"] == subset) & (model_subsets["model"] == model)
        ]
        acc_matrix[i, j] = row["Avg. acc_norm"].values[0]
        # print(row)
    # print(i, subset)

print(models)
print(acc_matrix.shape)
print(acc_matrix)
acc_matrix_ = acc_matrix.copy()

for i, subset in enumerate(subsets):
    # find the best and worst model
    max_acc_idx = np.argmax(acc_matrix[i, :])
    max_acc = acc_matrix[i, max_acc_idx]
    min_acc_idx = np.argmin(acc_matrix[i, :])
    min_acc = acc_matrix[i, min_acc_idx]

    # find the second best model
    acc_matrix[i, max_acc_idx] = -1
    second_max_acc_idx = np.argmax(acc_matrix[i, :])
    second_max_acc = acc_matrix[i, second_max_acc_idx]

    if max_acc - second_max_acc > 0.04:
        experts.append((i, subset, models[max_acc_idx]))

print(experts)
acc_matrix = acc_matrix_.copy()

for j, model in enumerate(models):
    if j == 0 or j == len(models) - 1:
        continue
    for jj in range(j + 1, len(models)):
        model_acc = acc_matrix[:, j]
        model_acc_ = acc_matrix_[:, jj]
        # print(model_acc, model_acc_)
        diff = model_acc - model_acc_
        max_diff = np.max(diff)
        min_diff = np.min(diff)

        print(f"model {jj} vs model {j}: {max_diff-min_diff}, {max_diff}, {min_diff}")
