### convert arm results: csv files to np arrays
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

dataset_size = 25256
dfs = {}
for model in bandits.values():
    csv_path = "./synced_data/csv/mmlu/airoboros-l2-70b_nochoice.csv"
    df = pd.read_csv(csv_path)
    dfs[model] = df
    print(model, df.shape)

### save inference results (y) to np arrays
acc_np = []
for idx_row in range(dataset_size):
    acc_arms = []
    for model in bandits.values():
        row = dfs[model].iloc[idx_row, :]
        acc = row["acc_norm"]
        acc_arms.append(acc)
    acc_np.append(acc_arms)

acc_np = np.array(acc_np)
print(acc_np.shape)
np.save("./synced_data/csv/mmlu/models_accnorm.npy", acc_np)
