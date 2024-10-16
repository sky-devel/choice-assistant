from transformers import AutoTokenizer
from torch import cuda, bfloat16
from time import time
import transformers
import torch

model_id = "./models/Vikhr-Llama-3.2-1B-instruct"
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)


time_1 = time()
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
)


query_pipeline = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)


while True:
    question = input("Enter a question: ")
    answer = query_pipeline(
        question,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=112,)
    print(answer, "\n")
