import os

import pandas as pd
from datasets import load_dataset

pd.options.display.float_format = "{:.2f}".format

models = [
    "gpt-4",
    "gpt-3.5-turbo",
]  # "claude-v1"] #, "claude-instant-v1"] #, "guanaco-33b"]

### check if file exists
tmp_file_path = "./synced_data/tmp.json"
csv_file_path = "./synced_data/chatbot_arena.csv"
if not os.path.exists(tmp_file_path):
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    rows = dataset.to_json(tmp_file_path, index=False)

### read json
battles = pd.read_json(tmp_file_path, lines=True)

questions = []
human_prefer = []
data_row = []
for idx, battle in battles.iterrows():
    conversation_a, conversation_b = battle["conversation_a"], battle["conversation_b"]
    q = []
    for conver in conversation_a:
        content = conver["content"]
        role = conver["role"]
        if role == "user":
            content = content.replace('"', "")
            content = content.replace("\n", "")
            content = content.replace("\r", "")
            content = content.replace("\t", "")
            q.append(content)
    q_ = []
    for conver in conversation_b:
        content = conver["content"]
        role = conver["role"]
        if role == "user":
            content = content.replace('"', "")
            content = content.replace("\n", "")
            content = content.replace("\r", "")
            content = content.replace("\t", "")
            q_.append(content)
    if q != q_:
        print("not equal")
        print(q)
        print(q_)
        input()
    q = " ".join(q)
    questions.append(q)

    winner_ab = battle["winner"]
    if winner_ab == "model_a" or winner_ab == "model_b":
        winner_model = battle[winner_ab]
        human_prefer.append(winner_model)
    elif winner_ab == "tie":
        winner_model = winner_ab
        human_prefer.append(winner_model)
    elif winner_ab == "tie (bothbad)":
        winner_model = winner_ab
        human_prefer.append(winner_model)
    else:
        print(winner_ab)

    data_row.append([q, winner_model, battle["model_a"], battle["model_b"]])

### save to csv
df = pd.DataFrame(data_row, columns=["question", "human_prefer", "model_a", "model_b"])
df.to_csv(csv_file_path, index=False, header=True, sep="|")

df = df[df["model_a"].isin(models) & df["model_b"].isin(models)]

print("Size of dataset after filtering by the models: ", len(df))
print(df.head())
print(df.tail())
print("Count of human prference: ", len(set(df["human_prefer"])))
print("Count of question: ", len(set(df["question"])))

### pivot the dataframe
qa_with_models = {}
for idx, row in df.iterrows():
    if row["question"] not in qa_with_models:
        qa_with_models[row["question"]] = [
            [row["human_prefer"], row["model_a"], row["model_b"]]
        ]
    else:
        qa_with_models[row["question"]].append(
            [row["human_prefer"], row["model_a"], row["model_b"]]
        )

print("Questions containing the models: ", len(qa_with_models))

cnt = 0
for q in qa_with_models:
    if len(qa_with_models[q]) != 1:
        # print(q)
        print(qa_with_models[q])
        cnt += 1

print(cnt)
