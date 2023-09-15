# from datasets import load_dataset
# dataset = load_dataset("open-llm-leaderboard/results", streaming=True) #, download_mode="force_redownload")
# dataset = load_dataset("open-llm-leaderboard/results", download_mode="force_redownload")

# dataset = load_dataset("open-llm-leaderboard/details_upstage__Llama-2-70b-instruct", 
#                             "harness_hellaswag_10",  split="latest")


from os import listdir
from os.path import isfile, join
import pandas as pd


path_list = [
            "synced_data/mmlu/details_upstage__Llama-2-70b-instruct",
            "synced_data/mmlu/details_upstage__Llama-2-70b-instruct-v2",
            "synced_data/mmlu/details_Aspik101__StableBeluga-13B-instruct-PL-lora_unload",
            "synced_data/mmlu/details_jondurbin__airoboros-l2-70b-gpt4-2.0",
            "synced_data/mmlu/details_Lajonbot__vicuna-7b-v1.5-PL-lora_unload"
            ]

path = './synced_data/mmlu/details_upstage__Llama-2-70b-instruct/'
model_name = 'Llama-2-70b-instruct'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
onlyfiles.sort()

mmlu_data = pd.DataFrame()
for file in onlyfiles:
    if file.endswith('.parquet'):
        # print(file)
        subdataset = file.split('|')[1]
        # print(subdataset)
        df = pd.read_parquet(path+"/"+file, engine='pyarrow')
        # select multiple columns
        df = df[['acc', 'acc_norm', 'choices', 'example', 'gold']]
        df["subdataset"] = subdataset
        df["model"] = model_name
        mmlu_data = pd.concat([mmlu_data, df])
        # print(mmlu_data.head())
        # print(mmlu_data.shape)

