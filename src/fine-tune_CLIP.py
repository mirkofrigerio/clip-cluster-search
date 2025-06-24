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


# === Load model & processor ===
MODEL_CHECKPOINT = "openai/clip-vit-base-patch32"
logging.info(f"Loading CLIP model and processor from {MODEL_CHECKPOINT}")

# Load the base CLIP model and processor
# Ensure this runs on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(MODEL_CHECKPOINT).to(device)
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
logging.info(f"Model loaded and moved to device: {device}")


def process_json_attributes(path):
    """
    Loads a gzipped JSON file, selects specific columns, and extracts text values
    from list-of-dictionary attributes, filtering by English language tags (starting with 'en').

    Args:
        path (str): The path to the gzipped JSON file.

    Returns:
        pd.DataFrame: A DataFrame with processed columns, where JSON attributes
                      are flattened into comma-separated strings, filtered by
                      any language tag that starts with 'en'.
    """
    try:
        df = pd.read_json(path, compression='gzip', lines=True)
    except Exception as e:
        logging.error(f"Error loading JSON file {path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

    columns_to_keep = ['item_id', 'main_image_id', 'item_name', 'bullet_point', 'brand', 'product_type', 'style', 'item_keywords']
    df = df.loc[:, columns_to_keep]

    for column in ['item_name', 'bullet_point' , 'brand', 'product_type', 'style', 'item_keywords']:
        df[column] = df[column].apply(
            lambda x: ', '.join(
                str(item.get('value'))
                for item in x
                if isinstance(item, dict)
                and 'value' in item
                and item.get('value') is not None
                and str(item.get('language_tag', '')).startswith('en')
            ) if isinstance(x, list) else ''
        )
    logging.info(f"Processed JSON attributes for file: {os.path.basename(path)}")
    return df

def concatenate_text_attributes(df):
    """
    Concatenate the values of each column into a single natural language string per row.
    Example: "the item_name is Shirt, the brand is Nike, the style is Casual"

    Args:
        df (pd.DataFrame): DataFrame containing product attributes.

    Returns:
        pd.DataFrame: DataFrame with 'attributes_nl' column and relevant IDs.
    """
    # Exclude item_id and main_image_id from concatenation
    columns = [col for col in df.columns if col not in ('item_id', 'main_image_id')]

    # Define a character limit for each attribute value to keep the concatenated string concise
    MAX_CHAR_LENGTH_PER_ATTRIBUTE = 50

    def row_to_natural_language(row):
        parts = []
        for col in columns:
            value = row[col]
            # Only include non-empty string values. Convert to string to avoid issues with non-string types.
            if isinstance(value, str) and value.strip():
                # Apply the character limit to the value
                limited_value = value.strip()
                if len(limited_value) > MAX_CHAR_LENGTH_PER_ATTRIBUTE:
                    limited_value = limited_value[:MAX_CHAR_LENGTH_PER_ATTRIBUTE] + "..." # Add ellipsis for truncated text
                parts.append(f"the {col.replace('_', ' ')} is {limited_value}")
        return ', '.join(parts)

    df['attributes_nl'] = df.apply(row_to_natural_language, axis=1)
    logging.info("Concatenated text attributes into 'attributes_nl' column.")
    return df[['item_id', 'main_image_id' ,'attributes_nl']]


def preprocess_data(examples, processor): # Added processor as argument
    """
    Preprocesses the data (text tokenization and image feature extraction)
    using the CLIP processor.

    Args:
        examples (dict): A dictionary containing 'text' (list of strings)
                         and 'image' (list of PIL Image objects).
        processor: The CLIP processor object.

    Returns:
        dict: A dictionary with 'input_ids', 'attention_mask', and 'pixel_values'
              ready for the model.
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


# --- Custom CLIP Trainer for Contrastive Loss ---
class CLIPTrainer(Trainer):
    """
    We have to create a custome trainer, since the standard Trainer expects the model to compute the loss internally when a forward pass is done.
    This is not the behaviour of CLIP, which, when given image and text, computes contrastive loss. Custom Trainer class for CLIP fine-tuning, specifically designed to handle
    the contrastive loss inherent to CLIP models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model # Ensure the model is available

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the contrastive loss for CLIP.
        The CLIP model's forward pass typically returns image_embeds, text_embeds, and logits_per_image/text.
        The loss is then computed from these logits.
        """
        # Ensure inputs are moved to the correct device
        # Filter inputs to only include what CLIPModel.forward() expects
        expected_keys = {'input_ids', 'attention_mask', 'pixel_values'}
        filtered_inputs = {k: v.to(model.device) for k, v in inputs.items() if k in expected_keys}
        
        outputs = model(**filtered_inputs, return_loss=True)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """
        Custom evaluation loop to calculate recall@k metrics typical for CLIP.
        """
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        all_image_embeds = []
        all_text_embeds = []
        
        total_loss = 0.0
        num_batches = 0

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            # forward pass for each input batch of the validation set, to compute loss and embeddings similarity.
            with torch.no_grad():
                outputs = model(**inputs, return_loss=True)
                loss = outputs.loss.item()
                total_loss += loss
                num_batches += 1

                # Get embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                all_image_embeds.append(image_embeds.cpu().numpy())
                all_text_embeds.append(text_embeds.cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        all_image_embeds = np.concatenate(all_image_embeds)
        all_text_embeds = np.concatenate(all_text_embeds)

        # Calculate cosine similarity matrix
        # (N_images, D) @ (D, N_texts) -> (N_images, N_texts)
        # remember, CLIP's loss isn't just cosine similarity, but it's an elaboration of that concept that introduces other params (temp)
        similarity_scores = all_image_embeds @ all_text_embeds.T

        # Calculate recall@k
        # recall here tells me how good the model is at retriving the right text given an image.
        # Assuming the diagonal elements are the correct pairs
        num_samples = similarity_scores.shape[0]

        k_values = [1, 5, 10] # Define k for recall@k
        recall_at_k = {f"recall@{k}": 0 for k in k_values}

        for i in range(num_samples):
            # Get indices of top_k similar texts for image i
            top_k_indices = np.argsort(similarity_scores[i, :])[::-1] # Descending order
            
            for k in k_values:
                if i in top_k_indices[:k]:
                    recall_at_k[f"recall@{k}"] += 1

        for k in k_values:
            recall_at_k[f"recall@{k}"] /= num_samples

        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        metrics.update({f"{metric_key_prefix}_{k}": v for k, v in recall_at_k.items()})

        # Log metrics
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics


class CLIPDataCollator:
    """
    Data Collator for CLIP.
    """
    def __init__(self, processor): # processor parameter is not strictly needed here if set_transform uses lambda
        self.processor = processor

    def __call__(self, features):
        pixel_values = torch.stack([item['pixel_values'].squeeze(0) for item in features])
        input_ids = torch.stack([item['input_ids'].squeeze(0) for item in features])
        attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in features])

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

# --- Start of the main execution block ---
if __name__ == '__main__':
    logging.info("Starting CLIP fine-tuning script.")
    ### -- Data Loading and Preprocessing -- ###

    logging.info("Processing JSON attributes from metadata files...")
    all_files = glob.glob(os.path.join(METADATA_DIR, '*.json.gz'))
    if not all_files:
        logging.warning(f"No .json.gz files found in {METADATA_DIR}. Exiting.")
        exit()

    dfs = []
    for path in all_files:
        df = process_json_attributes(path)
        if not df.empty:
            text_attr = concatenate_text_attributes(df)
            dfs.append(text_attr)

    text_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total processed text attributes rows: {len(text_df)}")

    logging.info("Loading image metadata...")
    img_df = pd.read_csv(IMAGE_CSV_PATH)
    img_df = img_df[['image_id', 'path']]
    img_df = img_df.rename(columns={'image_id': 'main_image_id'})
    logging.info(f"Total image metadata rows: {len(img_df)}")

    logging.info("Merging text attributes and image metadata...")
    merged_df = pd.merge(text_df, img_df, on='main_image_id', how='inner')
    
    # --- PATH CONSTRUCTION FOR S3 ---
    merged_df['image_path'] = merged_df['path'].apply(lambda x: f"{BASE_IMAGE_DIR}{x}")

    ### -- Hugging Face Dataset Creation and Preprocessing -- ###
    logging.info("Creating Hugging Face Dataset...")
    # datasets.Image() feature can handle S3 URIs if s3fs is installed and configured.
    dataset = Dataset.from_pandas(merged_df).cast_column("image_path", Image())

    dataset = dataset.rename_column("attributes_nl", "text")
    dataset = dataset.rename_column("image_path", "image")
    dataset = dataset.remove_columns(['item_id', 'main_image_id', 'path'])
    logging.info(f"Dataset created with {len(dataset)} examples.")
    logging.info("Applying preprocessing transformation...")
    # Pass processor to preprocess_data via lambda
    dataset.set_transform(lambda examples: preprocess_data(examples, processor)) 

    # Split dataset for training and validation
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    logging.info(f"Dataset split: Train {len(train_dataset)} examples, Eval {len(eval_dataset)} examples.")

    ### -- LoRA Configuration -- ###
    logging.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8, # Rank of the update matrices. Common values: 8, 16, 32, 64. Higher rank means more parameters, potentially better performance but less efficiency.
        lora_alpha=16, # LoRA scaling factor. Often set to 2 * r or r. Controls the scaling of the LoRA weights.
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1, # Dropout applied to the LoRA layers. Helps prevent overfitting.
        bias="none", # "none", "all", or "lora_only". "none" is usually fine.
        # task_type=TaskType.FEATURE_EXTRACTION, # Causing issues later
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    logging.info("LoRA applied to the model.")
    model.print_trainable_parameters()

    ### -- Training Setup -- ###
    data_collator = CLIPDataCollator(processor) # Use the custom collator

    logging.info("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=100,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=4,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    logging.info("Initializing Trainer...")
    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    ### -- Start Training -- ###
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed.")

    # Save the final LoRA adapters
    # Consider making this dynamic for Colab
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(FINAL_OUTPUT_DIR)
    logging.info(f"Final LoRA adapters saved to {FINAL_OUTPUT_DIR}")

    # Example of how to load and merge for inference (commented out)
    # from peft import PeftModel, PeftConfig
    # # Load the base model
    # base_model = AutoModel.from_pretrained(MODEL_CHECKPOINT).to(device)
    # # Load the PEFT model
    # peft_model = PeftModel.from_pretrained(base_model, final_output_dir)
    # # Merge LoRA weights into the base model (optional, for easier deployment)
    # merged_model = peft_model.merge_and_unload()
    # logging.info("LoRA adapters merged into the base model for inference.")
    # # Save the merged model if desired
    # # merged_model.save_pretrained("./clip_lora_merged_model")
