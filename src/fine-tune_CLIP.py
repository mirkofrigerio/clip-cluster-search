import glob
import os
import torch
import numpy as np
import pandas as pd
import logging

from transformers import AutoModel, AutoProcessor, TrainingArguments, Trainer
from datasets import Dataset, Image
from peft import LoraConfig, get_peft_model
from PIL import Image as PILImage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### -- Config:  Data Loading and Preprocessing -- ###
METADATA_DIR = '/content/drive/MyDrive/clip_project_data/listings/metadata/'
IMAGE_CSV_PATH ='s3://amazon-berkeley-objects/images/metadata/images.csv.gz'
BASE_IMAGE_DIR = 's3://amazon-berkeley-objects/images/small/'
OUTPUT_DIR = '/content/drive/MyDrive/clip_project_data/clip_lora_finetuned/'
FINAL_OUTPUT_DIR = '/content/drive/MyDrive/clip_project_data/clip_lora_finetuned_final_adapters/'
SAVED_EVAL_DATASET_DIR = '/content/drive/MyDrive/clip_project_data/eval_dataset_saved/'

### -- Config:  Model and Training Hyperparameters
MODEL_CHECKPOINT = "openai/clip-vit-base-patch32"
MAX_CHAR_LENGTH_PER_ATTRIBUTE = 100 # For concatenating text attributes
TEST_SPLIT_RATIO = 0.1 # Percentage of dataset to use for validation
RANDOM_SEED = 42
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128 # Matching train batch size for consistency, but eval not done during training
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
LOGGING_STEPS = 100 # Log training progress (loss, LR, etc.) every X steps
DATALOADER_NUM_WORKERS = 4 # Number of subprocesses for data loading

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
)

# Ensure this runs on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(MODEL_CHECKPOINT).to(device)
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
logging.info(f"Model loaded and moved to device: {device}")


# --- Helper Functions (Data Processing) ---

def process_json_attributes(path: str) -> pd.DataFrame:
    """
    Loads a gzipped JSON file, selects relevant columns, and extracts/flattens text values.
    Filters by English language tags.
    """
    try:
        df = pd.read_json(path, compression='gzip', lines=True)
    except Exception as e:
        logging.error(f"Error loading JSON file {path}: {e}")
        return pd.DataFrame()

    columns_to_keep = ['item_id', 'main_image_id', 'item_name', 'bullet_point', 'brand', 'product_type', 'style', 'item_keywords']
    df = df.loc[:, columns_to_keep]

    for column in ['item_name', 'bullet_point', 'brand', 'product_type', 'style', 'item_keywords']:
        df[column] = df[column].apply(
            lambda x: ', '.join(
                str(item.get('value'))
                for item in x
                if isinstance(item, dict) and 'value' in item and item.get('value') is not None
                and str(item.get('language_tag', '')).startswith('en')
            ) if isinstance(x, list) else ''
        )
    logging.info(f"Processed JSON attributes for file: {os.path.basename(path)}")
    return df

def concatenate_text_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate column values into a single natural language string per row, with truncation. 
    Note: The string becomes too long if we concatenate all the attributes. Here we limit item length in each to MAX_CHAR_LENGTH_PER_ATTRIBUTE
    """
    columns_to_concat = [col for col in df.columns if col not in ('item_id', 'main_image_id')]

    def row_to_natural_language(row) -> str:
        parts = []
        for col in columns_to_concat:
            value = row[col]
            if isinstance(value, str) and value.strip():
                limited_value = value.strip()
                if len(limited_value) > MAX_CHAR_LENGTH_PER_ATTRIBUTE:
                    limited_value = limited_value[:MAX_CHAR_LENGTH_PER_ATTRIBUTE] + "..."
                parts.append(f"the {col.replace('_', ' ')} is {limited_value}")
        return ', '.join(parts)

    df['attributes_nl'] = df.apply(row_to_natural_language, axis=1)
    logging.info("Concatenated text attributes into 'attributes_nl' column.")
    return df[['item_id', 'main_image_id', 'attributes_nl']]

def preprocess_data(examples: dict, processor: AutoProcessor) -> dict:
    """
    Preprocesses a batch of data (text tokenization and image feature extraction)
    using the provided CLIP processor. Designed for `Dataset.set_transform()`.
    """
    images = [img.convert("RGB") for img in examples['image']]
    texts = examples['text']

    image_inputs = processor.image_processor(
        images=images,
        return_tensors="pt"
    )

    text_inputs = processor.tokenizer(
        text=texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    return {
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'pixel_values': image_inputs['pixel_values'],
    }


# --- Custom Classes (Trainer and Data Collator) ---

class CLIPDataCollator:
    """
    Data Collator for CLIP. Stacks preprocessed tensors into batches.
    `processor` is not needed here as `preprocess_data` handles it.
    """
    def __call__(self, features: list[dict]) -> dict:
        pixel_values = torch.stack([item['pixel_values'].squeeze(0) for item in features])
        input_ids = torch.stack([item['input_ids'].squeeze(0) for item in features])
        attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in features])

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

class CLIPTrainer(Trainer):
    """
    Custom Trainer for CLIP fine-tuning, handling contrastive loss and evaluation.
    """
    def compute_loss(self, model: torch.nn.Module, inputs: dict, return_outputs: bool = False, num_items_in_batch: int = None) -> torch.Tensor:
        """Computes the contrastive loss for CLIP.
        The CLIP model's forward pass typically returns image_embeds, text_embeds, and logits_per_image/text. 
        The loss is then computed from these logits.
        """
        expected_keys = {'input_ids', 'attention_mask', 'pixel_values'}
        filtered_inputs = {k: v.to(model.device) for k, v in inputs.items() if k in expected_keys}
        
        outputs = model(**filtered_inputs, return_loss=True)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys: list[str] = None,
        metric_key_prefix: str = "eval",
    ) -> dict:
        """
        Custom evaluation loop for recall@k metrics.
        This evaluation loop is NOT used during `trainer.train()` if `eval_strategy="no"`.
        It would be called manually via `trainer.evaluate()` for post-training analysis.
        """
        model = self._wrap_model(self.model, training=False)
        model.eval()

        all_image_embeds = []
        all_text_embeds = []
        total_loss = 0.0
        num_batches = 0
        
        for step, inputs in enumerate(dataloader):
            expected_keys = {'input_ids', 'attention_mask', 'pixel_values'}
            inputs = {k: v.to(model.device) for k, v in inputs.items() if k in expected_keys}
            # forward pass for each input batch of the validation set, to compute loss and embeddings similarity.
            with torch.no_grad():
                outputs = model(**inputs, return_loss=True)
                loss = outputs.loss.item()
                total_loss += loss
                num_batches += 1

                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                all_image_embeds.append(image_embeds.cpu().numpy())
                all_text_embeds.append(text_embeds.cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if len(all_image_embeds) == 0:
            logging.warning("No image embeddings collected during evaluation. Returning only loss.")
            return {f"{metric_key_prefix}_loss": avg_loss}

        all_image_embeds = np.concatenate(all_image_embeds)
        all_text_embeds = np.concatenate(all_text_embeds)

        # Calculate cosine similarity matrix
        # (N_images, D) @ (D, N_texts) -> (N_images, N_texts)
        # remember, CLIP's loss isn't just cosine similarity, but it's an elaboration of that concept that introduces other params (temp)
        similarity_scores = all_image_embeds @ all_text_embeds.T

        num_samples = similarity_scores.shape[0]
        k_values = [1, 5, 10]
        recall_at_k = {f"recall@{k}": 0 for k in k_values}

        for i in range(num_samples):
            top_k_indices = np.argsort(similarity_scores[i, :])[::-1] # Descending order of similarity
            
            for k_val in k_values:
                if i in top_k_indices[:k_val]:
                    recall_at_k[f"recall@{k_val}"] += 1

        for k_val in k_values:
            recall_at_k[f"recall@{k_val}"] /= num_samples

        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        metrics.update({f"{metric_key_prefix}_{k}": v for k, v in recall_at_k.items()})

        logging.info(f"Evaluation metrics: {metrics}")
        return metrics


# --- Main Execution Block ---
if __name__ == '__main__':
    logging.info("Starting CLIP fine-tuning script.")

    logging.info("Processing JSON attributes from metadata files...")
    all_json_files = glob.glob(os.path.join(METADATA_DIR, '*.json.gz'))
    if not all_json_files:
        logging.warning(f"No .json.gz files found in {METADATA_DIR}. Exiting.")
        exit()

    dfs = []
    for path in all_json_files:
        df = process_json_attributes(path)
        if not df.empty:
            text_attr = concatenate_text_attributes(df)
            dfs.append(text_attr)

    text_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total processed text attributes rows: {len(text_df)}")

    logging.info(f"Loading image metadata CSV..")
    img_df = pd.read_csv(IMAGE_CSV_PATH)
    img_df = img_df[['image_id', 'path']].rename(columns={'image_id': 'main_image_id'})
    logging.info(f"Total image metadata rows: {len(img_df)}")

    logging.info("Merging text attributes and image metadata...")
    merged_df = pd.merge(text_df, img_df, on='main_image_id', how='inner')
    
    # Construct full S3 image URIs
    merged_df['image_path'] = merged_df['path'].apply(lambda x: f"{BASE_IMAGE_DIR}{x}")

    ### -- Hugging Face Dataset Creation and Preprocessing --
    logging.info("Creating Hugging Face Dataset...")
    dataset = Dataset.from_pandas(merged_df).cast_column("image_path", Image())

    dataset = dataset.rename_columns({"attributes_nl": "text", 
                                      "image_path": "image"})
    dataset = dataset.remove_columns(['item_id', 'main_image_id', 'path'])
    logging.info(f"Dataset created with {len(dataset)} examples.")

    # Train test split
    train_test_split = dataset.train_test_split(test_size=TEST_SPLIT_RATIO, seed=RANDOM_SEED)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test'] # Keep eval_dataset for potential manual evaluation later
    logging.info(f"Dataset split: Train {len(train_dataset)} examples, Eval {len(eval_dataset)} examples.")

    # --- Save raw eval_dataset to Google Drive ---
    os.makedirs(SAVED_EVAL_DATASET_DIR, exist_ok=True)
    eval_dataset.save_to_disk(SAVED_EVAL_DATASET_DIR)
    logging.info(f"Evaluation dataset saved to: {SAVED_EVAL_DATASET_DIR}")
    
    logging.info("Applying preprocessing transformation...")
    train_dataset.set_transform(lambda examples: preprocess_data(examples, processor)) 

    # 3. LoRA Configuration & Application
    logging.info("Applying LoRA configuration to the model...")
    model = get_peft_model(model, LORA_CONFIG)
    logging.info("LoRA applied to the model successfully.")
    model.print_trainable_parameters()

    # 4. Training Setup (Trainer and Arguments)
    data_collator = CLIPDataCollator() # Corrected: No processor needed here

    logging.info("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE, # Still provided, but not used during training due to eval_strategy="no"
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=100, # Saves a checkpoint every 100 steps
        report_to="wandb",
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        eval_strategy="no", # Explicitly disables evaluation during trainer.train()
        load_best_model_at_end=False, # Must be False if eval_strategy is "no"
    )

    logging.info("Initializing Trainer...")
    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 5. Start Training
    logging.info("Starting training process...")
    trainer.train()

    # Optional: Logic to resume from the latest checkpoint if it exists
    # This assumes OUTPUT_DIR is a persistent location (e.g., Google Drive)
    # last_checkpoint = None
    # if os.path.exists(OUTPUT_DIR):
    #     checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')]
    #     if checkpoints:
    #         latest_checkpoint_dir = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    #         last_checkpoint = os.path.join(OUTPUT_DIR, latest_checkpoint_dir)
    #         logging.info(f"Found existing checkpoint: {last_checkpoint}. Resuming training.")
    #     else:
    #         logging.info("No checkpoints found in output directory. Starting training from scratch.")
    # else:
    #     logging.info("Output directory does not exist. Starting training from scratch.")
    #     os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists if starting fresh

    # trainer.train(resume_from_checkpoint=last_checkpoint)
    logging.info("Training completed.")

    # 6. Save Final LoRA Adapters
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(FINAL_OUTPUT_DIR)
    logging.info(f"Final LoRA adapters saved to {FINAL_OUTPUT_DIR}")