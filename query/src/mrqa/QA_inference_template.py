### modified from https://huggingface.co/docs/transformers/tasks/question_answering, https://huggingface.co/deepset/roberta-base-squad2
### Example of inference on SQuAD v2.0 dataset: https://github.com/huggingface/datasets/blob/main/metrics/squad_v2/squad_v2.py
##### predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
##### references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
##### squad_v2_metric = datasets.load_metric("squad_v2")
##### results = squad_v2_metric.compute(predictions=predictions, references=references)
##### print(results)

### Just disables the warning, doesn't take advantage of AVX/FMA to run faster
### warning: To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import evaluate
from datasets import load_dataset
from transformers import pipeline

dataset_size = 100

### Load the validation set
val_set = load_dataset("squad_v2")["validation"].select(
    range(dataset_size)
)  ### id, title, context, question, answers, then samples. size=11873
model_name = "deepset/deberta-v3-base-squad2"

### Load the metric and pipeline
metric = evaluate.load("mrqa")  # squad_v2
question_answerer = pipeline(
    "question-answering", model=model_name, tokenizer=model_name
)

### Get predictions

### takes question and context as input
outputs = question_answerer(
    question=val_set["question"], context=val_set["context"]
)  ### returns samples, then answer, score, start, end

pred_answer = [
    {
        "id": val_set["id"][i],
        "prediction_text": outputs[i]["answer"] if outputs[i]["score"] >= 0.9 else "",
        "no_answer_probability": outputs[i]["score"],
    }
    for i in range(len(outputs))
]
groud_truth = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_set]
# print("pred_answer: ", pred_answer)
# print("groud_truth: ", groud_truth)
# print("model outputs: ", outputs)

# metric.add(predictions=answer, references=answer) or batched metric.add_batch(references=refs, predictions=preds)

### evaluate metrics on the validation set https://huggingface.co/spaces/evaluate-metric/squad_v2
"""
Expected format: {'predictions': {'id': Value(dtype='string', id=None), 'prediction_text': Value(dtype='string', id=None), 
                        'no_answer_probability': Value(dtype='float32', id=None)}, 
                'references': {'id': Value(dtype='string', id=None), 
                    'answers': Sequence(feature={'text': Value(dtype='string', id=None), 
                    'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None)}}
"""
scores = metric.compute(predictions=pred_answer, references=groud_truth)
print("scores: ", scores)
print(pred_answer[:10], groud_truth[:10])
print(len(pred_answer), len(groud_truth))


### Alternative with PyTorch
# import torch
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# inputs = tokenizer(QA_input["question"], QA_input["context"], return_tensors="pt")

# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# with torch.no_grad():
#     outputs = model(**inputs)

# ### Get the highest probability from the model output for the start and end positions
# answer_start_index = outputs.start_logits.argmax()
# answer_end_index = outputs.end_logits.argmax()

# ### Decode the predicted tokens to get the answer
# predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
# print(tokenizer.decode(predict_answer_tokens))
