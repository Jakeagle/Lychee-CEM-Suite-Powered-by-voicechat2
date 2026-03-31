import os
import json
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# --- CONFIGURATION ---
CHARACTER = "bones"
BASE_MODEL = "unsloth/meta-llama-3.1-8b-bnb-4bit" # Optimized for 8GB VRAM
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "voicechat2", "data", "characters", CHARACTER)
OUTPUT_DIR = os.path.join(DATA_DIR, "training_output")

print(f"--- 🦿 Lychee Forge: Training {CHARACTER} ---")

# 1. LOAD AND MERGE DATA
def prepare_dataset():
    combined_data = []
    data_files = ["lore.json", "conversational.json", "prose.json", "social.json"]
    
    for file_name in data_files:
        path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                content = json.load(f)
                combined_data.extend(content)
                print(f"Loaded {len(content)} entries from {file_name}")

    # Format for Llama 3.1 Instruct
    formatted = []
    for item in combined_data:
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['output']}<|eot_id|>"
        formatted.append({"text": text})
    
    return Dataset.from_list(formatted)

dataset = prepare_dataset()

# 2. LOAD MODEL
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 3. ADD LORA ADAPTERS (The "Personality Layer")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 4. RUN TRAINING
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Increased for better personality stickiness
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_is_bf16_supported(),
        bf16 = torch.cuda.is_is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

print("Starting training loop...")
trainer.train()

# 5. SAVE FOR OLLAMA
print("Training complete. Exporting GGUF...")
# This will overwrite your existing bones_final.gguf with the upgraded version
model.save_pretrained_gguf(os.path.join(HOME, "bones_final"), tokenizer, quantization_method = "q4_k_m")

print(f"--- ✅ SUCCESS: {CHARACTER} has been upgraded! ---")