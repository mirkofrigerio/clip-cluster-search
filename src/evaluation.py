 # --- Manual Evaluation Example (to be run separately after training, or in a new Colab session) ---
    # To evaluate the final saved model or a specific checkpoint:
    # IMPORTANT: You will need to re-initialize `processor` and `device` if running in a new session.

    # For example:
    import torch
    from transformers import AutoModel, AutoProcessor, TrainingArguments
    from datasets import Dataset, Image
    from peft import PeftModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor_eval = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    base_model_eval = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # 1. Load the saved eval_dataset (which is in its raw form)
    from datasets import Dataset
    loaded_eval_dataset = Dataset.load_from_disk(SAVED_EVAL_DATASET_DIR)
    logging.info(f"Loaded evaluation dataset from: {SAVED_EVAL_DATASET_DIR}")

    # 2. Apply the preprocessing transform to the loaded eval dataset
    loaded_eval_dataset.set_transform(lambda examples: preprocess_data(examples, processor_eval))

    # 3. Load the trained LoRA adapters onto the base model
    peft_model_eval = PeftModel.from_pretrained(base_model_eval, FINAL_OUTPUT_DIR)
    # Or to load a specific checkpoint:
    peft_model_eval = PeftModel.from_pretrained(base_model_eval, os.path.join(OUTPUT_DIR, "checkpoint-XXXX"))
    
    # 4. Create a new Trainer instance for evaluation only
    eval_trainer = CLIPTrainer(
       model=peft_model_eval,
       args=TrainingArguments(output_dir="./eval_results", per_device_eval_batch_size=EVAL_BATCH_SIZE),
       eval_dataset=loaded_eval_dataset, # Pass the transformed loaded eval_dataset
       data_collator=CLIPDataCollator(),
    )
    # 5. Run evaluation
    metrics = eval_trainer.evaluate()
    print(f"Manual Evaluation Metrics: {metrics}")
