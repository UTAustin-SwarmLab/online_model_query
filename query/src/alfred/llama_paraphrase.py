import time

import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline

model = "meta-llama/Llama-2-13b-chat-hf"
# device = "cuda:2" if torch.cuda.is_available() else "cpu"
aug_numbers = 7

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model)
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    temperature=0.7,
    do_sample=True,
    # device=device,
)
end_time = time.time()
print(f"loading model --- {end_time - start_time} seconds ---")

### load the alfred csv file
alfred_data = pd.read_csv(
    "./synced_data/csv/alfred_data/alfred_valid_language_goal.csv"
)
alfred_ans_data = pd.DataFrame(columns=alfred_data.columns)
start_time = time.time()
for idx, row in alfred_data.iterrows():
    if idx % 50 == 0:
        end_time = time.time()
        print(idx, f"--- {end_time - start_time} seconds ---")
        start_time = time.time()

    if row["lang_goal"][-1] != ".":
        row["lang_goal"] += "."
    question = (
        "Rewrite the following sentence: "
        + row["lang_goal"]
        + "\n"
        + "Answer in one sentence: "
    )
    # print(question, "\n--")
    sequences = gen_pipeline(
        question,
        do_sample=True,
        top_k=10,
        num_return_sequences=aug_numbers,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.99,
        max_length=100,
    )

    if row["lang_instr"][-1] != ".":
        row["lang_instr"] += "."
    instruct = (
        "Rewrite the following sentences: "
        + row["lang_instr"]
        + "\n"
        + "Answer in a three to five sentences: "
    )
    # print(instruct, "\n-----------------------------------")
    instruct_sequences = gen_pipeline(
        instruct,
        do_sample=True,
        top_k=10,
        num_return_sequences=aug_numbers,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.99,
        max_length=300,
    )

    for i, seq in enumerate(sequences):
        answer = seq["generated_text"].split(question)[1].strip().split(".")[0]
        # print(answer)
        # print("--")
        instruct_answer = (
            instruct_sequences[i]["generated_text"]
            .split(instruct)[1]
            .replace("\n", "")
            .strip()
        )
        # print(repr(instruct_answer))
        # print("-----------------------------------")
        alfred_ans_data.loc[len(alfred_ans_data)] = row
        alfred_ans_data.loc[len(alfred_ans_data) - 1, "lang_goal"] = answer
        alfred_ans_data.loc[len(alfred_ans_data) - 1, "lang_instr"] = instruct_answer
        alfred_ans_data.loc[len(alfred_ans_data) - 1, "repeat_idx"] = row[
            "repeat_idx"
        ] + 10 * (i + 1)
        # print(i, alfred_ans_data.loc[len(alfred_ans_data) - 1])
    # break

alfred_ans_data.to_csv(
    "./synced_data/csv/alfred_data/alfred_aug_valid_language_goal.csv", index=False
)
alfred_ans_data = pd.concat([alfred_data, alfred_ans_data])
alfred_ans_data.to_csv(
    "./synced_data/csv/alfred_data/alfred_merged_valid_language_goal.csv", index=False
)

"""
Paraphrase the following sentence: Place a cool apple in the white bin Answer in one sentence:  
-----------------------------------

Deposit a refreshing apple into the crisp white container.
Position a refreshing apple within the crisp white container.
Place a cool apple in the bin with a white lid.
Position a fresh apple on the white container.
"""
