### Test Vicuna-7b model with arbitrary question
import torch
from transformers import AutoTokenizer, pipeline

model = "lmsys/vicuna-7b-v1.5"  # "meta-llama/Llama-2-13b-chat-hf"
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

print(gen_pipeline.device)

question = """What is the source of the material that causes meteor showers?
A. Near-Earth asteroids gradually disintegrate and spread out along their orbital path. When the Earth passes through 
the orbit of an asteroid we are bombarded by sand-sized particles which cause a meteor shower. B. Near-Earth asteroids 
disintegrate as they enter Earth's atmosphere creating hundreds of bright meteors that appear to radiate from a single 
location in the sky. C. The nuclei of comets disintigrate as they enter Earth's atmosphere creating hundreds of bright 
meteors that appear to radiate from a central location in the sky. D. The nuclei of comets gradually disintegrate and 
spread out along the comet's orbital path. When the Earth passes through the orbit of a comet we are bombarded by 
sand-sized particles which cause a meteor shower."""

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
    answer = seq["generated_text"].split(question)[1]
    print(answer)

"""
Result: Question: Cities control the amount of pollution that is allowed to come from 
cars. How does this most likely help people? Answer: This helps people by reducing the 
amount of pollution that is released into the air, which can improve air quality and 
reduce the negative health effects associated with exposure to pollution. 
Additionally, reducing pollution from cars can also help to mitigate the effects of 
climate change by reducing greenhouse gas emissions.
"""
