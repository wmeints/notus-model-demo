from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "argilla/notus-7b-v1"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextIteratorStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=torch.bfloat16,
)

messages = [
    {
        "role": "system",
        "content": "You're a friendly assistant. You help people generate content for their blog.",
    },
    {
        "role": "user",
        "content": "Please generate an outline for a blogpost about using open source lanugage models",
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)

generator_thread = Thread(
    target=model.generate,
    kwargs=dict(
        inputs=inputs,
        max_new_tokens=2500,
        do_sample=True,
        temperature=0.7,
        streamer=streamer,
        top_p=0.98,
    ),
)

generator_thread.start()

for fragment in streamer:
    print(fragment, end="")
