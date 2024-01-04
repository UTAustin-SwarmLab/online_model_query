import argparse
import pickle

import clip
import nltk
import numpy as np
import pandas as pd
import torch
import torchvision
from clip.simple_tokenizer import SimpleTokenizer

torchvision.__version__

parser = argparse.ArgumentParser(description="Create Configuration")
parser.add_argument("-d", "--device", type=int, help="device name", default=-1)
args = parser.parse_args()
print(args)

### download nltk resources
nltk.download("punkt")

batch_size = 128
device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)

### Load the csv file
alfred_lang_data = (
    pd.read_csv("./synced_data/csv/alfred_data/alfred_merged_valid_language_goal.csv")
    .set_index(["task_idx", "repeat_idx"])
    .sort_index()
)
alfred_lang_data.fillna(value="", inplace=True)

### Load CLIP model
model, preprocess = clip.load(
    "ViT-L/14@336px", jit=False, device=device
)  # ViT-L/14@336px, ViT-B/32
tokenizer = SimpleTokenizer()

cnt = 0
instr_dict = {}
low_instr_dict = {}
instr_len = []
low_instr_len = []

### returns samples, then answer, score, start, end
for idx, row in alfred_lang_data.iterrows():
    if cnt % 500 == 0:
        print(cnt)
    cnt += 1

    ### encode the text
    task_idx = idx[0]
    repeat_idx = idx[1]
    instr = row["lang_goal"].replace('"', "")
    low_instr = row["lang_instr"].replace('"', "")
    low_instr_length = len(low_instr)

    low_instr_1 = low_instr[: int(low_instr_length / 3)]
    low_instr_2 = low_instr[int(low_instr_length / 3) : int(low_instr_length / 3) * 2]
    low_instr_3 = low_instr[int(low_instr_length / 3) * 2 :]

    instr_length = len(tokenizer.encode(instr))
    instr_len.append(instr_length)
    low_instr_length = len(tokenizer.encode(low_instr))
    low_instr_len.append(low_instr_length)

    token = clip.tokenize(
        [
            instr,
            low_instr_1,
            low_instr_2,
            low_instr_3,
        ],
        truncate=True,
    ).to(
        device
    )  # shape = [4, 77]
    emb = model.encode_text(token)  # shape = [4, 768]

    instr_dict[(task_idx, repeat_idx)] = emb[0].cpu().detach().numpy()
    low_instr_dict[(task_idx, repeat_idx)] = emb[1:4].flatten().cpu().detach().numpy()

## save dictionaries
print(len(instr_dict))
pickle.dump(
    instr_dict, open("./synced_data/csv/alfred_data/clip_emb_instruct.pkl", "wb")
)

print(len(low_instr_dict))
pickle.dump(
    low_instr_dict,
    open("./synced_data/csv/alfred_data/clip_emb_low_instruct.pkl", "wb"),
)

### save instruction length
instr_len = np.array(instr_len)
print(instr_len.shape)
np.save("./synced_data/csv/alfred_data/instruct_token_length.npy", instr_len)

low_instr_len = np.array(low_instr_len)
print(low_instr_len.shape)
np.save("./synced_data/csv/alfred_data/low_instruct_token_length.npy", low_instr_len)
