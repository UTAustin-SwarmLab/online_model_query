### convert MMLU data to csv file
import json
import os
from os import listdir
from os.path import isfile, join

import pandas as pd

path_list = [
    "synced_data/mmlu/details_tiiuae__falcon-180B",
    # "synced_data/mmlu/details_tiiuae__falcon-180B-chat",
    # "synced_data/mmlu/details_augtoma__qCammel-70-x",
    # "synced_data/mmlu/details_upstage__Llama-2-70b-instruct",
    # "synced_data/mmlu/details_upstage__Llama-2-70b-instruct-v2",
    # "synced_data/mmlu/details_Aspik101__StableBeluga-13B-instruct-PL-lora_unload",
    # "synced_data/mmlu/details_jondurbin__airoboros-l2-70b-gpt4-2.0",
    # "synced_data/mmlu/details_Lajonbot__vicuna-7b-v1.5-PL-lora_unload",
]

model_name_list = [
    "falcon-180B",
    # "falcon-180B-chat",
    # "qCammel-70-x",
    # "Llama-2-70b-instruct",
    # "Llama-2-70b-instruct-v2",
    # "StableBeluga-13B",
    # "airoboros-l2-70b",
    # "vicuna-7b-v1.5",
]

subdatasets = []

include_example = False
df_goal_shape = (25256, 6 + int(include_example))

for path, model_name in zip(path_list, model_name_list):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles.sort()

    mmlu_data = pd.DataFrame()
    for file in onlyfiles:
        if file.endswith(".parquet") and "truthfulqa:mc" not in file:
            subdataset = file.split("|")[1]
            print(file)
            df = pd.read_parquet(path + "/" + file, engine="pyarrow")
            if include_example:
                df = df[["acc", "acc_norm", "example", "choices", "gold"]]
            else:
                df = df[["acc", "acc_norm", "example", "gold"]]
            df["example"] = df["example"].apply(lambda x: x.split("Answer:")[0].strip())
            df["subdataset"] = subdataset
            subdatasets.append(subdataset)
            df["model"] = model_name
            mmlu_data = pd.concat([mmlu_data, df])

    print(mmlu_data.shape)
    assert mmlu_data.shape == df_goal_shape, f"shape is not {df_goal_shape}"

    ### save to csv file
    csv_path = "./synced_data/csv/mmlu/"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if include_example:
        mmlu_data.to_csv(csv_path + f"{model_name}.csv", index=True)
    else:
        mmlu_data.to_csv(csv_path + f"{model_name}_nochoice.csv", index=True)

json_dict = {i: subdataset for i, subdataset in enumerate(subdatasets)}
print(json_dict)
with open("synced_data/mmlu/subdatasets.json", "w") as outfile:
    json.dump(json_dict, outfile, indent="\t")


# from datasets import load_dataset
# dataset = load_dataset("open-llm-leaderboard/results", streaming=True) #, download_mode="force_redownload")
# dataset = load_dataset("open-llm-leaderboard/results", download_mode="force_redownload")

# dataset = load_dataset("open-llm-leaderboard/details_upstage__Llama-2-70b-instruct",
#                             "harness_hellaswag_10",  split="latest")