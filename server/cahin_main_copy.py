import transformers

# RAG
# import faiss
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document



from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch
from constants import *
import os
import numpy as np
import json
import re



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
            s = ""
            f.write(s)
    
    ## if no rag file, create rag file
    
    rag_name = os.path.join(RAG_SAVE_PATH, "RAG.json")
    return file_name, msg_name , rag_name


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

def add_to_last_RAG_json(data, data_type, msg, time="time", id=1):
    print(data['history'].keys())
    if data_type == "constants":
        data["Constants"][time][f"id_{id}"] = msg
    elif data_type == "history":
        data["history"][time][f"id_{id}"] = msg
    return data

def write_to_RAG_json(rag_name, data):
    # data.encode('utf-8', errors='ignore').decode('utf-8')
    # data = json.dumps(data, indent=4)
    
    with open(rag_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# def check_CONSTANTS_in

def convert_messages_to_prompt(message, response_type="one_sentanse"):
    """
    Convert a list of message dictionaries to a formatted prompt string.
    
    Args:
    messages (list): A list of dictionaries with 'role' and 'content' keys
    
    Returns:
    str: Formatted prompt string
    """
    role = f"<|start_header_id|>{message['role']}<|end_header_id|>"
    end_prompt = "<|eot_id|>"
    assistant_start_prompt = "<|start_header_id|>assistant<|end_header_id|>"
    formatted_prompt = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nutting Knowledge Date: December 2023\n\n''' + end_prompt

    message_list = [x for x in message['content'].split("\n")]
    result_string = "".join(message_list) + end_prompt

    return_message_list = []
    for m in message_list:
        return_message_list.append(
            role
            + m
            + "<|eot_id|>"
        )


    if response_type == "one_sentanse":      # for response

        if message['role'] == "user":
            return role + '\n\n' + result_string + assistant_start_prompt + '\n\n'
            
        if message['role'] == "assistant":
            return result_string

    elif response_type == "multi_sentanse":  # constants, for response
        return formatted_prompt + role + '\n\n' + result_string 

    elif response_type == "first_constants": # constants, for rag
        
        return  return_message_list




if __name__ == "__main__":
    file_name, msg_name, rag_name = check_file()
    print("Chat name: ", file_name, msg_name)
    if READ_FROM_MEMORY is not None:
        messages = list(np.load(msg_name, allow_pickle=True))
    else:
        messages = [{"role": "system", "content": ""}]
        messages.append(
                    {"role": "user", "content": SYSTEM_PROMPT}
                )


    print(f"Using model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


    hf = HuggingFacePipeline.from_model_id(
        model_id=MODEL_ID,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 1024,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "return_full_text": False,  # 只返回生成的文本
            "clean_up_tokenization_spaces": True
        },
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "cuda",
        }
    )
    hf.pipeline.tokenizer.pad_token_id = hf.pipeline.tokenizer.eos_token_id

    # embedding function
    embedding_function = HuggingFaceBgeEmbeddings(
        # model_name = "BAAI/bge-large-zh",
        model_name = "BAAI/bge-m3",
        model_kwargs = {'device':'cuda'},
        encode_kwargs = {'normalize_embeddings':True}
    )


    if not os.path.exists(rag_name):

        ini_rag_data = {"Constants":{
                                "time":{}
                     }, "history":{
                                "time":{}
        }}
        with open(rag_name, 'w', encoding='utf-8') as f:
            json.dump(ini_rag_data, f, ensure_ascii=False, indent=4)
    
    
    rag_history = read_from_RAG_json(rag_name)
    new_conversation_id = len(rag_history["history"]["time"])   
    # rag_history = add_to_last_RAG_json(rag_history, "history", ["1","2"], id=new_conversation_id)

    
    new_conversation = []

    rag_system_prompt = convert_messages_to_prompt({"role": "user", "content": INI_RAG}, response_type="first_constants")
    system_prompt = convert_messages_to_prompt({"role": "user", "content": SYSTEM_PROMPT}, response_type="multi_sentanse")

    rag_history = add_to_last_RAG_json(rag_history, "constants", rag_system_prompt)
    write_to_RAG_json(rag_name, rag_history)
    
    constant_loader = JSONLoader(
         file_path=rag_name,
         jq_schema=f'.Constants.time.id_{1}',
         text_content=False,)
    rag_input = constant_loader.load()
    all_history_list = constant_loader.load()[0].page_content[2:-2].split("', '")
    

    # for i in range(0, len(rag_history["history"]["time"])):
    #     history_loader = JSONLoader(
    #         file_path=rag_name,
    #         jq_schema=f'.history.time.id_{i}',
    #         text_content=False,)
    #     all_history_list = all_history_list + history_loader.load()[0].page_content[2:-2].split("', '")

    rag_input[0].page_content = str(all_history_list) #[1:-1]
    
    docs = []
    for doc in all_history_list:

        docs.append(Document(metadata=rag_input[0].metadata, page_content=doc))


    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)


    rag_prompt = ""
    # main loop
    while True:
        # # input msg
        # input_P = str(input("Enter your message: "))
        # if input_P == "bye":
        #     break
        input_P = "實驗室"
        

        # align user msg to llm input type
        user_one_msg = {"role": "user", "content": input_P}
        
        messages.append(user_one_msg)
        prompt = convert_messages_to_prompt(user_one_msg, response_type="one_sentanse")
        # docs = text_splitter.split_documents(documents=rag_input)
        # print(docs)
        # exit()

        # vector store
        db = Chroma.from_documents(docs, embedding_function) #, persist_directory="./rag_save/rag_db/first_db.db")


        try:
            mq_retriever = MultiQueryRetriever.from_llm(retriever = db.as_retriever(), llm = hf)
            retrieved_docs = mq_retriever.get_relevant_documents(query=input_P)

            # retriever = db.as_retriever()
            # retrieved_docs = retriever.get_relevant_documents(query=input_P, top_k=20)
        except:
            print("Bad input: ", user_one_msg["content"])
            messages.pop(-1)
            continue

        print("----"*20)
        for doc in retrieved_docs:
            print(doc.page_content)
        print("----"*20)

        # embeddings_filter  = EmbeddingsFilter(embeddings=embedding_function, similarity_threshold=0.4)
        # compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
        # compressed_docs = compression_retriever.get_relevant_documents(query = input_P)
        # print(compressed_docs[0].page_content)

        
        def check_new_history(rag_prompt, related_history_content):
            related_history = re.search(r"<\|end_header_id\|>(.*?)<\|eot_id\|>", related_history_content).group(1)
            if related_history:
                if rag_prompt.find(related_history) == -1:
                    print("Find new history")
                    rag_prompt += related_history
            return rag_prompt

        for retrieved_one in retrieved_docs:
            ## check if the new related history is already in the 'rag_prompt' if True, then skip
            rag_prompt = check_new_history(rag_prompt, retrieved_one.page_content)
        
        
        # final_input = rag_prompt + system_prompt + "".join(new_conversation) + prompt
        print(f"system_prompt: {system_prompt}")
        print("----"*20)
        print(f"rag_system_prompt: {rag_system_prompt}")
        print("----"*20)
        print(f"prompt: {prompt}")
        print("----"*20)
        print(f"rag_prompt: {rag_prompt}")
        exit()
        final_input = system_prompt + "".join(new_conversation) + prompt
        print("----"*20)
        print("final_input: ", final_input)
        # AI response
        try:
            out_msg = hf.invoke(final_input)

        except:
            print("Bad input: ", user_one_msg["content"])
            messages.pop(-1)
            continue
        
        # align AI msg to llm input type
        assistant_one_msg = {"role": "assistant", "content": out_msg}

        # write to chat and rag and msg(for llm input)
        write_to_chat(file_name, user_one_msg["role"], user_one_msg["content"])
        # write to chat and rag 
        write_to_chat(file_name, assistant_one_msg['role'], assistant_one_msg['content'])
        messages.append(assistant_one_msg)

        # save chat and messages(for next llm input)
        np.save(msg_name, messages)

        assistant_prompt = convert_messages_to_prompt(assistant_one_msg, response_type="one_sentanse")

        new_conversation.append(prompt)
        new_conversation.append(assistant_prompt)


        # rag_history = add_to_last_RAG_json(rag_history, "history", new_conversation, id=new_conversation_id)
        # write_to_RAG_json(rag_name, rag_history)

        show_chat(file_name)
        prompt_tokens = len(prompt)
        print(prompt_tokens)
        print("\n")

    print("Goodbye!")
