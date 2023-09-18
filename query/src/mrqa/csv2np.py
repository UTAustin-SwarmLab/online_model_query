### convert arm results: csv files to np arrays
import numpy as np
import pandas as pd

bandits = {
    0: "deberta-v3-base-mrqa",
    1: "deberta-v3-base-squad2",
    2: "bigbird-base-trivia-itc",
    3: "distilbert-base-uncased-distilled-squad",
    4: "roberta-base-squad2-nq",
}

datasets = [
    "SQuAD",
    "TriviaQA-web",
    "NaturalQuestionsShort",
    "NewsQA",
    "SearchQA",
    "HotpotQA",
]
dataset_size = (10507, 7785, 12836, 4212, 16980, 5901)

dfs = {}
for model in bandits.values():
    for dataset in datasets:
        csv_path = f"./synced_data/csv/mrqa/{dataset}_{model}_validation.csv"
        df = pd.read_csv(csv_path)
        dfs[(dataset, model)] = df
        print(dataset, model, df.shape)

### save inference results (y) to np arrays
for idx_ds, dataset in enumerate(datasets):
    size = dataset_size[idx_ds]
    f1_np = []
    exact_np = []
    for idx_row in range(size):
        f1_arms = []
        exact_arms = []
        for model in bandits.values():
            row = dfs[(dataset, model)].iloc[idx_row, :]
            f1 = row["f1"]
            exact = row["exact"]
            f1_arms.append(f1)
            exact_arms.append(exact)
        f1_np.append(f1_arms)
        exact_np.append(exact_arms)

    f1_np = np.array(f1_np)
    exact_np = np.array(exact_np)
    print(f1_np.shape)
    print(exact_np.shape)
    np.save(f"./synced_data/csv/mrqa/{dataset}_f1.npy", f1_np)
    np.save(f"./synced_data/csv/mrqa/{dataset}_exact.npy", exact_np)
