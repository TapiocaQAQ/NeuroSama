import transformers
import torch
from constants import *


generation_prompt = AI_NAME + ": "
host_prompt = HOST_NAME + ": "

# model_id = "FlagAlpha/Llama3-Chinese-8B-Instruct"
model_id = "yentinglin/Llama-3-Taiwan-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)

terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

messages = [{"role": "system", "content": ""}]

messages.append(
                {"role": "user", "content": SYSTEM_PROMPT_CH}
            )

while True:
    chat_prompt = input("Enter your message: ")
    
    if chat_prompt == "bye":
        break

    # add user message
    messages.append(
        {"role": "user", "content": chat_prompt}
    )

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    content = outputs[0]["generated_text"][len(prompt):]
    print(content)
    print("----"*30)
    print(outputs)
    print("----"*30)

    # add assistant message
    messages.append(
        {"role": "assistant", "content": content}
    )



