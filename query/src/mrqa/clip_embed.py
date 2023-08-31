### python ./src/mrqa/clip_embed.py -d 0 -ds NaturalQuestionsShort
import argparse

import clip
import nltk
import numpy as np
import torch
import torchvision
from datasets import load_dataset
from nltk.tokenize import sent_tokenize

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
args = parser.parse_args()
print(args)

### download nltk resources
nltk.download("punkt")

dataset = args.dataset
batch_size = 128
device = (
    f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
)

### Load the validation set
val_set = load_dataset("mrqa", split="validation", streaming=False, use_auth_token=True)
### select subset of the validation set: 'SQuAD', 'HotpotQA', 'TriviaQA-web',
### 'NaturalQuestionsShort', 'SearchQA', 'NewsQA'
val_set = val_set.filter(
    lambda example: example["subset"] == dataset
)  ### id, title, context, question, answers, then samples. size=11873
print("val_set: ", val_set)
print("dataset: ", dataset, "model_name: ", "clip")

### Load CLIP model
model, preprocess = clip.load("ViT-B/32", jit=False, device=device)

idx = 0
q_list = []
c_list = []
a_list = []

### returns samples, then answer, score, start, end
for example in val_set:
    if idx % 200 == 0:
        print(idx)

    ### encode the text
    question = example["question"].replace("<P>", "").replace("</P>", "")
    context = example["context"].replace("<P>", "").replace("</P>", "")
    context = sent_tokenize(context, language="english")
    context1 = " ".join(context[: len(context) // 2])
    context2 = " ".join(context[len(context) // 2 :])
    answer = example["answers"][0]

    token = clip.tokenize([question, context1, context2, answer], truncate=True).to(
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
np.save(f"./csv/mrqa/clip_emb_{dataset}_question.npy", q_list)

c_list = np.array(c_list)
print(c_list.shape)
np.save(f"./csv/mrqa/clip_emb_{dataset}_context.npy", c_list)

a_list = np.array(a_list)
print(a_list.shape)
np.save(f"./csv/mrqa/clip_emb_{dataset}_answer.npy", a_list)
