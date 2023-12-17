import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "argilla/notus-7b-v1"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
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

output = model.generate(
    inputs=inputs,
    max_new_tokens=2500,
    do_sample=True,
    temperature=0.7,
    top_p=0.98,
)

response_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(response_text)
