import torch
import transformers
from transformers import AutoTokenizer

model = "meta-llama/Llama-2-13b-chat-hf"
device = "cuda:3" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    # device_map="auto",
    device=device,
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
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

### Conversation
# from transformers import Conversation, pipeline  # noqa: F401

# model = # "facebook/m2m100_418M"

# text_generator = pipeline("text-generation", model="meta-llama/Llama-2-13b-hf")
# text2text_generator = pipeline("text-generation")
# _ = text2text_generator(
#     "Hello, I'm a language model", max_length=30, num_return_sequences=3
# )
# print(_[0]["generated_text"])  ## Hello, I'm a language model (yes-yes-yes). So, what do you mean?

# text2text_generator = pipeline(
#     "text2text-generation", model="meta-llama/Llama-2-7b-chat-hf"
# )
# _ = text2text_generator(
#     "question: What is 42 ? context: 42 is the answer to life, the universe and everything"
# )
# print(_[0]["generated_text"])
# _ = text2text_generator("translate from English to French: I'm very happy")
# print(_[0]["generated_text"]) ## Je suis très heureux
# converse = pipeline("conversational", model="meta-llama/Llama-2-7b-chat-hf")
# conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
# conversation_2 = Conversation("What's the last book you have read?")
# answer = converse([conversation_1, conversation_2])
# print(answer)
# print(type(answer))  # list
# print(len(answer))  # 2
# print(type(answer[0]))
# """
# user >> Going to the movies tonight - any suggestions?
# bot >> The Big Lebowski
# , Conversation id: a6b4b4b6-77e2-4fd5-a055-f542dd99c319
# user >> What's the last book you have read?
# bot >> The Last Question
# """


### using model tokenizer
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

# text = "I like to play video games"
# input = tokenizer.encode(text, return_tensors="pt")
# outputs = model.generate(input, max_length=100, num_return_sequences=5, temperature=0.1)
# for i, output in enumerate(outputs):
#     print(f"{i}: {tokenizer.decode(output)}")
