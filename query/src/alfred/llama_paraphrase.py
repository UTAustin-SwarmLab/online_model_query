import time

import torch
import transformers
from transformers import AutoTokenizer

model = "meta-llama/Llama-2-13b-chat-hf"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    temperature=0.1,
    torch_dtype="auto",
    # device_map="auto",
    device=device,
)
end_time = time.time()
print(f"loading llama model --- {end_time - start_time} seconds ---")

prompt = "Paraphrase the following sentence:"
q = "Place a cool apple in the white bin"
question = f"Question: {prompt} {q}\nAnswer:"

start_time = time.time()
sequences = pipeline(
    question,
    do_sample=True,
    top_k=50,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=50,
)
end_time = time.time()
print(f"Generating answer --- {end_time - start_time} seconds ---")

for seq in sequences:
    print(seq["generated_text"])
    output = seq["generated_text"]
