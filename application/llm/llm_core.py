from transformers import AutoTokenizer
from torch import cuda, bfloat16
from langchain.prompts import PromptTemplate
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

template = PromptTemplate(
    template="""
        Answer the question in two sentences: {question}
     """,
    input_variables=["question"],
)

while True:
    question = input("Enter your question: ")
    answer = query_pipeline(
        template.format(question=question),
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=512,
    )
    print(answer[0]["generated_text"], "\n")
