from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig
import torch 

model_name = "/home/asif/llm/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto",load_in_4bit=True)
model.config.use_cache = False
dataset_name = "LinhDuong/chatdoctor-200k"
dataset = load_dataset(dataset_name)
# data = load_dataset("timdettmers/openassistant-guanaco")
# data = dataset.shuffle().map(generate_prompt)
# print(data)
training_args = TrainingArguments(
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 16,
    num_train_epochs=5,
    learning_rate=2e-5,
    save_total_limit=4,
    logging_steps=1,
    output_dir="/home/asif/llm/llama-2-chat-medi",
    max_steps=10000,
    gradient_checkpointing = True,
    optim="adamw_torch",
    save_strategy = "epoch",
    lr_scheduler_type="cosine",
    fp16=True,
    warmup_ratio = 0.05,
    report_to = 'tensorboard'
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"### Question: {example['input'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    max_seq_length=512,
    formatting_func=formatting_prompts_func,
    args=training_args
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32) 
        
trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model 
model_to_save.save_pretrained("llama-2-chat-medical") 