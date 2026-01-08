import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "distilgpt2"
DATA_PATH = "data/instructions.json"
OUTPUT_DIR = "./lora-output"


with open(DATA_PATH, "r") as f:
    data = json.load(f)

texts = [
    f"Instruction: {item['instruction']}\nAnswer: {item['output']}"
    for item in data
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(text):
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )


dataset = [tokenize(t) for t in texts]


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn"],  
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)