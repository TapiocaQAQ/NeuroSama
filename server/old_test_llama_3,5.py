from transformers import AutoModel, AutoTokenizer
from constants import *
from PIL import Image


generation_prompt = AI_NAME + ": "
host_prompt = HOST_NAME + ": "
model = AutoModel.from_pretrained('yentinglin/Llama-3-Taiwan-8B-Instruct', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('yentinglin/Llama-3-Taiwan-8B-Instruct', trust_remote_code=True)
# model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()

chat_prompt = "Hello Luna, how are you doing today?"




image = Image.open('/home/user/Workspace/NeuroSama/server/jjchen.jpg').convert('RGB')

msgs = []

full_prompt = SYSTEM_PROMPT 

while True:
    chat_prompt = input("Enter your message: ")
    
    if chat_prompt == "bye":
        break
    full_prompt += host_prompt + chat_prompt + '\n' + generation_prompt
    
    
    msgs = [{'role': "user", 'content': full_prompt}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )

    print(res+'\n')

    full_prompt += res + '\n'

    print(msgs[0])
    print("\n")
    
    prompt_tokens = len(tokenizer.apply_chat_template(msgs, tokenize=True, return_tensors="pt")[0])
    print(prompt_tokens)
    print("\n\n")




# return {
#     "mode": "instruct",
#     "stream": True,
#     "max_tokens": 200,
#     "skip_special_tokens": False,  # Necessary for Llama 3
#     "custom_token_bans": BANNED_TOKENS,
#     "stop": STOP_STRINGS,
#     "messages": [{
#         "role": "user",
#         "content": self.generate_prompt()
#     }]
# }