import argparse

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
parser.add_argument(
    "-ds", "--dataset", type=str, help="dataset name", default="fractal20220817_data"
)
args = parser.parse_args()
print(args)

### download nltk resources
nltk.download("punkt")

batch_size = 128
device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)

### Load the csv file
rtx_lang_data = pd.read_csv(f"./synced_data/rtx/{args.dataset}_small.csv")
rtx_lang_data.fillna(value="", inplace=True)

### Load CLIP model
model, preprocess = clip.load(
    "ViT-L/14@336px", jit=False, device=device
)  # ViT-L/14@336px, ViT-B/32
tokenizer = SimpleTokenizer()

cnt = 0
instr_np = []
instr_length_list = []

### returns samples, then answer, score, start, end
for idx, row in rtx_lang_data.iterrows():
    if cnt % 500 == 0:
        print(cnt)
    cnt += 1

    ### encode the text
    instr = row["text"]
    instr_length = len(tokenizer.encode(instr))
    instr_length_list.append(instr_length)

    token = clip.tokenize(
        instr,
        truncate=True,
    ).to(
        device
    )  # shape = [1, 77]
    emb = model.encode_text(token)  # shape = [1, 768]
    instr_np.append(emb.cpu().detach().numpy().reshape(-1))

### save instruction length
instr_np = np.array(instr_np)
print(instr_np.shape)
np.save(f"./synced_data/csv/rtx/{args.dataset}_instruct_emb.npy", instr_np)

### save instruction length
instr_length_list = np.array(instr_length_list)
print(instr_length_list.shape)
np.save(f"./synced_data/csv/rtx/{args.dataset}_instruct_length.npy", instr_length_list)
