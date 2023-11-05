### This script is used to generate answers for MMLU questions
import os  # noqa: F401
import time
import warnings

import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline  # noqa: F401

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
warnings.filterwarnings("ignore")
# model = "lmsys/vicuna-7b-v1.5"
# model = "jondurbin/airoboros-l2-70b-3.1.2"
# model = "stabilityai/StableBeluga-13B"
# model = "upstage/Llama-2-70b-instruct"
model = "tiiuae/falcon-180B-chat"

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

time_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model)
AutoTokenizer.from_pretrained(model)
# gen_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     temperature=0.1,
#     do_sample=True,
#     # device=device,
# )
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

time_end = time.time()

laod_time = time_end - time_start
print("load time: ", laod_time)

### generate questions
mmlu_data = pd.read_csv("./synced_data/csv/mmlu/" + "vicuna-7b-v1.5" + ".csv")
answer_time = []
for idx, row in mmlu_data.iterrows():
    time_start = time.time()
    question = row["example"] + " Answer in short: "

    sequences = gen_pipeline(
        question,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )

    answer = sequences[0]["generated_text"].split(question)[1].strip()
    time_end = time.time()
    print(idx, time_end - time_start)
    answer_time.append(time_end - time_start)

    if (idx + 1) % 50 == 0:
        break

model = model.split("/")[1]
df = pd.DataFrame(answer_time, columns=["answer_time"])
df["load_time"] = laod_time
df["model"] = [model] * len(df)
df.to_csv(f"./synced_data/csv/mmlu/{model}_answer_time.csv", index=False)
