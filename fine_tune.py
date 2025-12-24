import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from typing import Dict, List

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "finetune_lora_multilingual_150k.jsonl"
OUTPUT_DIR = "./qwen2.5_lora"

MAX_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 2
LR = 2e-4
SEED = 42
DEBUG = False

torch.manual_seed(SEED)

# =========================
# LOAD DATASET
# =========================
print("📦 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

print(f"Loaded {len(dataset):,} samples from {DATA_PATH}")

# DEBUG MODE: very small subset
if DEBUG:
    print("⚠️  DEBUG MODE: Using only 100 samples!")
    dataset = dataset.shuffle(seed=42).select(range(100))
    EPOCHS = 10
    
dataset = dataset.train_test_split(test_size=0.1, seed=SEED)

print(f"Train samples: {len(dataset['train']):,}")
print(f"Test samples: {len(dataset['test']):,}")

# =========================
# TOKENIZER
# =========================
print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# IMPROVED PROMPT FORMATTING WITH LABEL MASKING
# =========================
def format_and_tokenize(example):
    """
    Format conversation and tokenize with robust label masking.
    Only assistant responses are used for loss calculation.
    
    This implementation avoids tokenization misalignment by:
    1. Formatting the full conversation first
    2. Tracking where assistant content starts
    3. Using return_offsets_mapping to find exact token boundaries
    """
    # Build conversation parts separately to track positions
    conversation_parts = []
    assistant_content_start_char = 0
    current_char_pos = 0
    
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            user_text = f"<|user|>\n{content}\n"
            conversation_parts.append(user_text)
            current_char_pos += len(user_text)
        else:  # assistant
            assistant_marker = "<|assistant|>\n"
            conversation_parts.append(assistant_marker)
            # Record where actual assistant content starts (after the marker)
            assistant_content_start_char = current_char_pos + len(assistant_marker)
            conversation_parts.append(f"{content}\n")
            current_char_pos += len(assistant_marker) + len(content) + 1
    
    # Combine all parts
    full_text = "".join(conversation_parts) + tokenizer.eos_token
    
    # Tokenize with offset mapping to get character-to-token alignment
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None,
        return_offsets_mapping=True  # This gives us char positions
    )
    
    # Create labels
    labels = tokenized["input_ids"].copy()
    
    # Use offset mapping to find where to start training
    # offset_mapping is a list of (start_char, end_char) tuples for each token
    offset_mapping = tokenized["offset_mapping"]
    
    # Mask all tokens that end before the assistant content starts
    for i, (start_char, end_char) in enumerate(offset_mapping):
        if end_char <= assistant_content_start_char:
            labels[i] = -100  # Ignore this token in loss calculation
    
    # Remove offset_mapping from the output (not needed for training)
    tokenized.pop("offset_mapping")
    tokenized["labels"] = labels
    
    return tokenized

# Apply formatting and tokenization
print("🔧 Formatting and tokenizing dataset...")
dataset = dataset.map(
    format_and_tokenize,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing",
    num_proc=4  # Parallel processing for speed
)

# =========================
# DATA COLLATOR
# =========================
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Optional, Union

@dataclass
class DataCollatorForSupervisedDataset:
    """
    Custom data collator that pads sequences dynamically
    and preserves label masking (-100 for ignored tokens)
    """
    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        
        # Pad sequences to the longest in the batch
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        
        for ids, lbls, mask in zip(input_ids, labels, attention_mask):
            padding_length = max_length - len(ids)
            
            # Pad input_ids with pad_token_id
            padded_input_ids.append(
                ids + [self.tokenizer.pad_token_id] * padding_length
            )
            
            # Pad labels with -100 (ignore index)
            padded_labels.append(
                lbls + [-100] * padding_length
            )
            
            # Pad attention_mask with 0
            padded_attention_mask.append(
                mask + [0] * padding_length
            )
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long)
        }

data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# =========================
# MODEL
# =========================
print("🤖 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,  # Updated from torch_dtype
    device_map="auto",
    trust_remote_code=True
)

# =========================
# LoRA
# =========================
print("🧩 Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("\n📊 Trainable parameters:")
model.print_trainable_parameters()

# =========================
# TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=50,
    
    # Save at end of each epoch
    save_strategy="epoch",
    save_steps=None,  # Disable step-based saving
    save_total_limit=2,  # Keep only last 2 checkpoints
    
    # Evaluate at end of each epoch
    eval_strategy="epoch",
    eval_steps=None,  # Disable step-based eval
    
    load_best_model_at_end=False,
    report_to="none",
    seed=SEED,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
)

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

# =========================
# TRAIN
# =========================
print(f"\n🚀 Starting training for {EPOCHS} epochs...")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"Total optimization steps: {len(dataset['train']) // (BATCH_SIZE * GRAD_ACCUM) * EPOCHS}")

trainer.train()

# =========================
# SAVE FINAL MODEL
# =========================
final_path = f"{OUTPUT_DIR}/final"
print(f"\n💾 Saving final model to {final_path}")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print("\n✅ Training complete!")
print(f"Models saved in: {OUTPUT_DIR}")
print(f"- Checkpoints: {OUTPUT_DIR}/checkpoint-*")
print(f"- Final model: {final_path}")