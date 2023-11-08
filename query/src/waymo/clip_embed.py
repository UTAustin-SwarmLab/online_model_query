### extract CLIP embeddings for each question, choices, and answer of vicuna-7b-v1.5
import argparse

import clip
import nltk
import numpy as np
import torch
import torchvision
from clip.simple_tokenizer import SimpleTokenizer

torchvision.__version__

prompt = "Answer whether or not there is a {} in the picture and where is it."
DETECT_OBJ = [
    "car",
    "truck",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "pole",
    "sign",
    "traffic_light",
    "building",
    "vegetation",
]
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


### Load CLIP model
model, preprocess = clip.load(
    "ViT-L/14@336px", jit=False, device=device
)  # ViT-L/14@336px, ViT-B/32

tokenizer = SimpleTokenizer()
q_list = []
q_len = []

### returns samples, then answer, score, start, end
for obj in DETECT_OBJ:
    ### encode the text
    question = prompt.format(obj)
    token = clip.tokenize([question], truncate=True).to(device)  # shape = [1, 77]
    input_length = len(tokenizer.encode(question))
    q_len.append(input_length)

    emb = model.encode_text(token)  # shape = [1, 768]
    q_list.append(emb[0].cpu().detach().numpy())

### save lists
q_list = np.array(q_list)
print(q_list.shape)
np.save("./synced_data/csv/waymo/clip_emb_question.npy", q_list)

q_len = np.array(q_len)
print(q_len.shape)
np.save("./synced_data/csv/waymo/question_token_length.npy", q_len)
