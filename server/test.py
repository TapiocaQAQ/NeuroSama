import transformers

# RAG
# import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch
from constants import *
import os
import numpy as np
import json


# check file
def check_file():
    # check if the chat folder exists
    if not os.path.exists(CHAT_SAVE_PATH):
        os.makedirs(CHAT_SAVE_PATH)
    if not os.path.exists(MSG_SAVE_PATH):
        os.makedirs(MSG_SAVE_PATH)
    if not os.path.exists(RAG_SAVE_PATH):
        os.makedirs(RAG_SAVE_PATH)

    # make a new chat or read from memory
    if len(os.listdir(CHAT_SAVE_PATH)) == 0:
        print("Creating frist chat...")
        file_name = os.path.join(CHAT_SAVE_PATH, f"chat_{str(0).zfill(3)}.txt")
        msg_name = os.path.join(MSG_SAVE_PATH, file_name.split("/")[-1].replace(".txt", ".npy"))
        

        with open(file_name, 'w') as f:
            s = "\nassistant:\n" + SYSTEM_PROMPT + "\n"
            f.write(s)

    elif READ_FROM_MEMORY is not None:
        file_name = os.path.join(CHAT_SAVE_PATH, READ_FROM_MEMORY)
        msg_name = os.path.join(MSG_SAVE_PATH, READ_FROM_MEMORY.split(".")[0] + ".npy")
        assert os.path.exists(file_name), f"Chat file {file_name} does not exist!"
        assert os.path.exists(msg_name), f"Msg file {msg_name} does not exist!"
        print(f"Reading from memory... {file_name}")

    else:
        last_file = sorted(os.listdir(CHAT_SAVE_PATH))[-1]
        file_name = os.path.join(CHAT_SAVE_PATH, last_file[:5] + str(int(last_file[5:8]) + 1).zfill(3) + ".txt")
        msg_name  = os.path.join(MSG_SAVE_PATH, file_name.split("/")[-1].replace(".txt", ".npy"))
        print(f"Creating new chat... {file_name}")
        with open(file_name, 'w') as f:
            s = "\nsystem:\n" + SYSTEM_PROMPT + "\n"
            f.write(s)

    # rag_name = os.path.join(RAG_SAVE_PATH, "RAG.json")
    return file_name, msg_name #, rag_name


# define write chat
def write_to_chat(file_name, role, input_msg):
    clean_input_msg = input_msg.encode('utf-8', errors='ignore').decode('utf-8')
    with open(file_name, 'a') as f:
        s = f"{role}: " + clean_input_msg + "\n"
        f.write(s)

# define read chat
def show_chat(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            if IS_CHANGE_ROLE:
                for key in CHANGE_ROLE:
                    line = line.strip().replace(key, CHANGE_ROLE[key])
            print(line)

def read_from_RAG_json(rag_name):
    with open(rag_name, 'r') as f:
        data = json.load(f)
    return data

def add_to_last_RAG_json(data, msg):
    
    last_key = data.keys()[-1]
    data[last_key+1] = {"text": msg["role"] + ": " + msg["content"]}
    return data

def write_to_RAG_json(rag_name, data):

    with open(rag_name, 'w') as f:
        json.dump(data, f) 


if __name__ == "__main__":
    file_name, msg_name = check_file()
    print("Chat name: ", file_name, msg_name)
    model_id = "./models/my_meta-llama/Llama-3.1-8B-Instruct"

    print(f"Using model: {model_id}")

    hf = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=0,
        batch_size=8
    )
    if READ_FROM_MEMORY is not None:
        messages = list(np.load(msg_name, allow_pickle=True))
    else:
        messages = [("system", "")]
        messages.append(
                    ("user", SYSTEM_PROMPT)
                )

    while True:

        if input_P == "bye":
            break
        input_P = str(input("Enter your message: "))
        user_one_msg = {"user", input_P}
        messages.append(user_one_msg)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是資深 Python 工程師"),
            ("user", f"{input_P}")
        ])
    
    chain = prompt | hf

    print(chain)


    

    print("Goodbye!")