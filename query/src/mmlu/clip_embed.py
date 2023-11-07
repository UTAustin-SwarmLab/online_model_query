### extract CLIP embeddings for each question, choices, and answer of vicuna-7b-v1.5
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
mmlu_data.fillna(value="No response.", inplace=True)

### Load CLIP model
model, preprocess = clip.load(
    "ViT-L/14@336px", jit=False, device=device
)  # ViT-L/14@336px, ViT-B/32
# print(clip.available_models())

tokenizer = SimpleTokenizer()

idx = 0
q_list = []
c_list = []
a_list = []
q_len = []
a_len = []

### returns samples, then answer, score, start, end
for idx, row in mmlu_data.iterrows():
    if idx % 500 == 0:
        print(idx)

    ### encode the text
    question = row["example"]
    choices = row["choices"].replace("[", "").replace("]", "").replace("'", "")
    answer = row["answer"].replace("[", "").replace("]", "").replace("\n", "")
    token = clip.tokenize([question, choices, answer], truncate=True).to(
        device
    )  # shape = [3, 77]

    input_length = len(tokenizer.encode(question + choices))
    q_len.append(input_length)
    answer_length = len(tokenizer.encode(answer))
    a_len.append(answer_length)

    emb = model.encode_text(token)  # shape = [3, 512]

    q_list.append(emb[0].cpu().detach().numpy())
    c_list.append(emb[1].flatten().cpu().detach().numpy())
    a_list.append(emb[2].cpu().detach().numpy())

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

q_len = np.array(q_len)
print(q_len.shape)
np.save("./synced_data/csv/mmlu/question_token_length.npy", q_len)

a_len = np.array(a_len)
print(a_len.shape)
np.save("./synced_data/csv/mmlu/answer_token_length.npy", a_len)
