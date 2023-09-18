### python ./src/mrqa/clip_embed.py -d 0 -ds NaturalQuestionsShort
import argparse

import clip
import nltk
import numpy as np
import pandas as pd
import torch
import torchvision

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
mmlu_data = pd.read_csv("./synced_data/csv/mmlu/vicuna-7b-v1.5_quest_ans.csv")

### Load CLIP model
model, preprocess = clip.load("ViT-B/32", jit=False, device=device)

idx = 0
q_list = []
c_list = []
a_list = []

### returns samples, then answer, score, start, end
for idx, row in mmlu_data.iterrows():
    if idx % 500 == 0:
        print(idx)

    ### encode the text
    question = row["example"]
    choices = row["choices"].replace("[", "").replace("]", "")
    answer = row["answers"]

    token = clip.tokenize([question, choices, answer], truncate=True).to(
        device
    )  # shape = [3, 77]
    emb = model.encode_text(token)  # shape = [3, 512]

    q_list.append(emb[0].cpu().detach().numpy())
    c_list.append(emb[1:3].flatten().cpu().detach().numpy())
    a_list.append(emb[-1].cpu().detach().numpy())

    idx += 1

### save lists
q_list = np.array(q_list)
print(q_list.shape)
np.save("./synced_data/csv/mmlu/clip_emb_question.npy", q_list)

c_list = np.array(c_list)
print(c_list.shape)
np.save("./synced_data/csv/mmlu/clip_emb_choices.npy", c_list)

a_list = np.array(a_list)
print(a_list.shape)
np.save("./synced_data/csv/mmlu/clip_emb_answer.npy", a_list)
