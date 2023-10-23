#### the code is not working now (cuda memory), need to be fixed

import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf").to(
    "cuda:2"
)

batch_size = 2
aug_numbers = 2

### load the alfred csv file
alfred_data = pd.read_csv(
    "./synced_data/csv/alfred_data/alfred_valid_language_goal.csv"
)
alfred_ans_data = pd.DataFrame(columns=alfred_data.columns)
start_time = time.time()

questions = []

for idx, row in alfred_data.iterrows():
    if row["lang_goal"][-1] != ".":
        row["lang_goal"] += "."
    question = (
        "Rewrite the following sentence: "
        + row["lang_goal"]
        + "\n"
        + "Answer in one sentence: "
    )
    if row["lang_instr"][-1] != ".":
        row["lang_instr"] += "."
    instruct = (
        "Rewrite the following sentences: "
        + row["lang_instr"]
        + "\n"
        + "Answer in a three to five sentences: "
    )
    questions.append(question)
    questions.append(instruct)

    if (idx + 1) % batch_size == 0:
        print(len(questions))
        number_sentences = int(len(questions) / 2)
        input_tokens = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(model.device)
        print(question, "\n--")
        sequences = model.generate(
            **input_tokens,
            min_length=64,
            max_length=64,
            do_sample=True,
            temperature=0.99,
            num_return_sequences=aug_numbers,
        )
        sequences = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        for i, seq in enumerate(sequences):
            if i >= number_sentences:
                break
            answer = seq.replace("\n", "").strip()
            print(answer)
            print("--")
            instruct_answer = sequences[i + number_sentences].replace("\n", "").strip()
            print(repr(instruct_answer))
            print("-----------------------------------")
            alfred_ans_data.loc[len(alfred_ans_data)] = row
            alfred_ans_data.loc[len(alfred_ans_data) - 1, "lang_goal"] = answer
            alfred_ans_data.loc[
                len(alfred_ans_data) - 1, "lang_instr"
            ] = instruct_answer
            alfred_ans_data.loc[len(alfred_ans_data) - 1, "repeat_idx"] = row[
                "repeat_idx"
            ] + 10 * (i + 1)
            print(i, alfred_ans_data.loc[len(alfred_ans_data) - 1])
        break

# alfred_ans_data.to_csv(
#     "./synced_data/csv/alfred_data/alfred_aug_batch_valid_language_goal.csv", index=False
# )
# alfred_ans_data = pd.concat([alfred_data, alfred_ans_data])
# alfred_ans_data.to_csv(
#     "./synced_data/csv/alfred_data/alfred_merged_batch_valid_language_goal.csv", index=False
# )
