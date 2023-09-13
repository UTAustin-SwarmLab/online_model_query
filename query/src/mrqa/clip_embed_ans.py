### python ./src/mrqa/clip_embed.py -d 0 -ds NaturalQuestionsShort
import argparse
import os

import clip
import nltk
import numpy as np
import pandas as pd
import torch
import torchvision

torchvision.__version__

parser = argparse.ArgumentParser(description="Create Configuration")
parser.add_argument("-d", "--device", type=int, help="device name", default=-1)
parser.add_argument(
    "-ds",
    "--dataset",
    type=str,
    help="sub dataset of mrqa",
    default="NaturalQuestionsShort",
)
parser.add_argument(
    "-mn",
    "--modelname",
    type=str,
    help="name of pre-trained model",
    default="distilbert-base-uncased-distilled-squad",
)
args = parser.parse_args()
print(args)

### download nltk resources
nltk.download("punkt")

dataset = args.dataset
batch_size = 128
device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)
csv_dir = "./synced_data/csv/mrqa/"
model_name = args.modelname
# VMware/deberta-v3-base-mrqa, deepset/deberta-v3-base-squad2, google/bigbird-base-trivia-itc
# distilbert-base-uncased-distilled-squad, nlpconnect/roberta-base-squad2-nq

### Load the csv file
os.makedirs(csv_dir, exist_ok=True)
if "/" in model_name:
    model_name = model_name.split("/", 1)[1]
df = pd.read_csv(csv_dir + f"{dataset}_{model_name}_validation.csv")


### Load CLIP model
model, preprocess = clip.load("ViT-B/32", jit=False, device=device)

### feed model's answer from csv file to numpy embeddings array
a_list = []
df.fillna(value="None", inplace=True)

for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(idx)

    ### encode the text
    pred_answer = row["pred_text"]
    # print(idx, pred_answer)
    token = clip.tokenize([pred_answer], truncate=True).to(device)  # shape = [1, 77]
    emb = model.encode_text(token)  # shape = [1, 512]
    a_list.append(emb[0].cpu().detach().numpy())

    idx += 1

### save lists
a_list = np.array(a_list)
print(a_list.shape)
np.save(
    f"./synced_data/csv/mrqa/clip_emb_{model_name}_{dataset}_predanswer.npy", a_list
)
