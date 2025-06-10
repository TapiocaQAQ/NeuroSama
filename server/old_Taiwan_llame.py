import transformers
import torch


model_id = "FlagAlpha/Llama3-Chinese-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)


messages = [{"role": "system", "content": ""}]

messages.append(
                {"role": "user", "content": "介绍一下机器学习"}
            )

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )


terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
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