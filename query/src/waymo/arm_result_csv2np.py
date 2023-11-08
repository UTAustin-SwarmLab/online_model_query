### convert arm results: csv files to np arrays
import numpy as np
import pandas as pd

bandits = {
    0: "llava-v1.5-7b",
    1: "llava-v1.5-13b",
    2: "llava-v1.5-13b-lora",
}

dataset_size = 20000
dfs = {}
for model in bandits.values():
    csv_path = f"./synced_data/csv/waymo/waymo_captions_{model}.csv"
    df = pd.read_csv(csv_path)
    dfs[model] = df
    print(model, df.shape)

### save inference results (y) to np arrays
acc_np = []
time_np = []
for idx_row in range(dataset_size):
    acc_arms = []
    time_arms = []
    for model in bandits.values():
        row = dfs[model].iloc[idx_row, :]
        time = row["inference time"]
        cnt = row["count of detecting object"]
        if cnt > 0:
            acc = 1 if "Yes" in str(row["caption"])[:3] else 0
        else:
            acc = 1 if "No" in str(row["caption"])[:2] else 0
        acc_arms.append(acc)
        time_arms.append(time)
    acc_np.append(acc_arms)
    time_np.append(time_arms)

acc_np = np.array(acc_np)
print(acc_np.shape)
np.save("./synced_data/csv/waymo/arm_results.npy", acc_np)
time_np = np.array(time_np)
print(time_np.shape)
np.save("./synced_data/csv/waymo/arm_results_time.npy", time_np)
