# Use a pipeline as a high-level helper
import torch
from transformers import pipeline
from transformers import AutoTokenizer

model = "lmsys/vicuna-7b-v1.5" # "meta-llama/Llama-2-13b-chat-hf"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model)
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    # device_map="auto",
    device=device,
)

question = "Question: Cities control the amount of pollution that is allowed to come from cars. How does this most likely help people? Answer:"

sequences = gen_pipeline(
    question,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

"""
Result: I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?
I'm also interested in documentaries, especially ones about history, science, and technology. Can you recommend some good ones?
Thanks!
"""
