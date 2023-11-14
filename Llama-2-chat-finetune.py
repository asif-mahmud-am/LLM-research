from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model

dataset_name = 'LinhDuong/chatdoctor-200k' 
dataset = load_dataset(dataset_name, split="train")

model_name = "/home/asif/llm/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    # load_in_8bit=True,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


lora_alpha = 16
lora_dropout = 0.5
lora_r = 32

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"### User: {example['input'][i]}\n ### Chatbot: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

output_dir = "./results_llama2_chat"
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 8
optim = "paged_adamw_32bit"
# optim="adamw_torch"
save_strategy = "steps"
save_steps = 500
logging_steps = 1
learning_rate = 2e-5
weight_decay = 0.
max_grad_norm = 0.3
max_steps = 10000
warmup_ratio = 0.03
lr_scheduler_type = "cosine"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_strategy=save_strategy,
    weight_decay=weight_decay,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    # dataset_text_field="text",
    formatting_func=formatting_prompts_func,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train() 

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model 
model_to_save.save_pretrained("llama-2-chat-medical") 