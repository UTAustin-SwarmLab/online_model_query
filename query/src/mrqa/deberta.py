import argparse
import os

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    default="deepset/deberta-v3-base-squad2",
)
args = parser.parse_args()
print(args)

dataset = args.dataset
batch_size = 128
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
model_name = args.modelname
# VMware/deberta-v3-base-mrqa, deepset/deberta-v3-base-squad2, google/bigbird-base-trivia-itc
# distilbert-base-uncased-distilled-squad, nlpconnect/roberta-base-squad2-nq

### Load the validation set
val_set = load_dataset("mrqa", split="validation", streaming=False, use_auth_token=True)
### select subset of the validation set: 'SQuAD', 'HotpotQA', 'TriviaQA-web'
### 'NaturalQuestionsShort',  'SearchQA', 'NewsQA'

val_set = val_set.filter(
    lambda example: example["subset"] == dataset
)  ### id, title, context, question, answers, then samples. size=11873
print("val_set: ", val_set)
print("dataset: ", dataset, "model_name: ", model_name)

### Load the metric and pipeline
metric = evaluate.load("squad_v2")
question_answerer = pipeline(
    "question-answering", model=model_name, tokenizer=model_name, device=device
)
validation_data = []

idx = 0
### returns samples, then answer, score, start, end
for outputs in question_answerer(
    question=val_set["question"], context=val_set["context"], batch_size=batch_size
):
    ### output: {'score': 0.9997267127037048, 'start': 158, 'end': 166, 'answer': ' France.'}

    if idx % 200 == 0:
        print(idx)

    ### Get predictions
    ### takes question and context as input
    id = val_set["qid"][idx]
    pred_text = outputs["answer"]
    score = outputs["score"]
    gold_answer_set = "&".join(val_set["detected_answers"][idx]["text"])
    # HasAns = False if len(val_set['answers'][idx]['answer_start']) == 0 else True
    # print("gold_answer_set: ", gold_answer_set)

    pred_answer = [
        {"id": id, "prediction_text": pred_text, "no_answer_probability": score}
    ]
    groud_truth = [
        {
            "id": val_set[idx]["qid"],
            "answers": {
                "text": val_set["detected_answers"][idx]["text"],
                "answer_start": [
                    0 for _ in range(len(val_set["detected_answers"][idx]["text"]))
                ],
            },
        }
    ]

    ### evaluate metrics on the validation set
    eval_score = metric.compute(predictions=pred_answer, references=groud_truth)
    # print("pred_answer: ", pred_answer, "groud_truth: ", groud_truth)
    # print("eval_score: ", eval_score)

    validation_data.append(
        [
            idx,
            id,
            pred_text,
            gold_answer_set,
            score,
            eval_score["f1"],
            eval_score["exact"],
            model_name,
        ]
    )

    idx += 1

df_validation = pd.DataFrame(
    validation_data,
    columns=[
        "idx",
        "id",
        "pred_text",
        "gold_answer_set",
        "score",
        "f1",
        "exact",
        "model",
    ],
)
df_validation.set_index("idx")

csv_dir = "./synced_data/csv/mrqa/"
os.makedirs(csv_dir, exist_ok=True)
if "/" in model_name:
    model_name = model_name.split("/", 1)[1]
df_validation.to_csv(csv_dir + f"{dataset}_{model_name}_validation.csv", index=False)
