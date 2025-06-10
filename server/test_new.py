import transformers

# RAG
# import faiss
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch
from constants import *
import os
import numpy as np
import json



model_id = "./models/my_meta-llama/Llama-3.1-8B-Instruct"
print(f"Using model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)


hf = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        # "return_full_text": False,  # 只返回生成的文本
        "clean_up_tokenization_spaces": True
    },
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
    }
)
    
# hf = HuggingFacePipeline.from_model_id(
#         model_id=model_id,
#         task="text-generation",
#         pipeline_kwargs={"max_new_tokens": 128,
#                          "eos_token_id": 2,
#                          "pad_token_id":2,
#                          "do_sample": True,
#                          "temperature": 0.6,
#                          "top_p": 0.9,
#         },
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device=0,
#         batch_size=8
#     )


print(hf.pipeline.tokenizer.eos_token_id)
# hf.pipeline.tokenizer.eos_token_id=2
# print(hf.pipeline.tokenizer.eos_token_id)
# exit()
hf.pipeline.tokenizer.pad_token_id = hf.pipeline.tokenizer.eos_token_id


# prompt = ChatPromptTemplate.from_messages([
#     ("system","你是一位很會教{topic}的老師."),
#     ("human","可以再說一次嗎?"),
#     ("ai","好的，我再講一次"),
#     ("human","{input}"),
# ])

# messages_list = prompt.format_messages(topic="數學",input="甚麼是三角函數?")

# user_one_msg = {"role": "user", "content": input_P}


# print(messages_list)

# print(hf.invoke(messages_list))

print(hf.invoke('''<|start_header_id|>user<|end_header_id|>你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''))
exit()







