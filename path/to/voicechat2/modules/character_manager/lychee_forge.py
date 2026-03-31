import os
import json
import torch
import sys
# --- 1. MEMORY FRAGMENTATION FIX ---
# This prevents the "CUDA out of memory" when small gaps exist in VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

if len(sys.argv) < 2:
    print("Error: No character name provided.")
    sys.exit(1)

CHARACTER = sys.argv[1]
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "voicechat2", "data", "characters", CHARACTER)
OUTPUT_GGUF = os.path.join(DATA_DIR, f"{CHARACTER}_brain") 

print(f"--- 🦿 Lychee Forge: Training {CHARACTER} [VRAM SAFE MODE] ---")

def prepare_dataset():
    combined_data = []
    data_files = ["lore.json", "conversational.json", "prose.json", "social.json"]
    for file_name in data_files:
        path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                content = json.load(f)
                combined_data.extend(content)
    
    formatted = []
    for item in combined_data:
        # Using a shorter template to save space
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['output']}<|eot_id|>"
        formatted.append({"text": text})
    return Dataset.from_list(formatted)

dataset = prepare_dataset()

# --- 2. SEQUENCE LENGTH TWEAK ---
# We are dropping this from 2048 to 1024. Bones's responses are short, 
# so 1024 is plenty and saves massive VRAM.
max_seq_length = 1024 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/meta-llama-3.1-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16, lora_dropout = 0, bias = "none",
)

# --- 3. BATCH SIZE REDUCTION ---
trainer = SFTTrainer(
    model = model, tokenizer = tokenizer, train_dataset = dataset,
    dataset_text_field = "text", max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1, # Dropped from 2 to 1 for 8GB safety
        gradient_accumulation_steps = 8, # Increased to keep effective batch size at 8
        warmup_steps = 5, 
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1, 
        output_dir = "outputs",
        # Use 8-bit optimizer to save even more VRAM
        optim = "adamw_8bit", 
    ),
)

trainer.train()
print(f"Exporting GGUF to {OUTPUT_GGUF}.gguf...")
model.save_pretrained_gguf(OUTPUT_GGUF, tokenizer, quantization_method = "q4_k_m")
print(f"--- ✅ SUCCESS: {CHARACTER} upgraded ---")