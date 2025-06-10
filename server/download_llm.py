import transformers
import torch
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# pipeline = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs={"torch_dtype": torch.float16},
#         device="cuda",
#         batch_size=8
#     )
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda",
    batch_size=8
)

pipeline.save_pretrained(f"./models/my_{model_id.split('/')[1]}")