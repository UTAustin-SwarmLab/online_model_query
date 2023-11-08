### This script is used to generate answers for MMLU questions
import warnings

import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline

warnings.filterwarnings("ignore")
# model = "lmsys/vicuna-7b-v1.5"  # "meta-llama/Llama-2-13b-chat-hf"
model = "jondurbin/airoboros-l2-70b-3.1.2"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model)
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    temperature=0.1,
    do_sample=True,
    # device=device,
)

### load the mmlu csv file

mmlu_data = pd.read_csv("./synced_data/csv/mmlu/" + "vicuna-7b-v1.5" + ".csv")
mmlu_quest_ans = pd.DataFrame(columns=["example", "choices", "answer"])
for idx, row in mmlu_data.iterrows():
    if idx % 500 == 0:
        print(idx)
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

    mmlu_quest_ans.loc[len(mmlu_quest_ans)] = {
        "example": row["example"],
        "answer": answer,
        "choices": row["choices"],
    }

mmlu_quest_ans.to_csv(
    "./synced_data/csv/mmlu/vicuna-7b-v1.5_quest_ans.csv", index=False
)

"""
Result: Question: Cities control the amount of pollution that is allowed to come from cars. How does this most likely 
help people? Answer: This helps people by reducing the amount of pollution that is released into the air, which can 
improve air quality and reduce the negative health effects associated with exposure to pollution. Additionally, reducing
pollution from cars can also help to mitigate the effects of climate change by reducing greenhouse gas emissions.
"""
