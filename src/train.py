"""
Training Script for MusicGen Baseline.

This script trains the MusicGen model using the FMA dataset pipeline.
"""

import argparse
from pathlib import Path
from src.data.pipeline import create_manifest_from_hf
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from audiocraft.models import MusicGenModel
from audiocraft.data import MusicDataset

def train_model(manifest_dir: Path, epochs: int, batch_size: int):
    print(f"Training model with data from {manifest_dir} for {epochs} epochs and batch size {batch_size}.")

    train_manifest = manifest_dir / "train/data.jsonl"
    valid_manifest = manifest_dir / "valid/data.jsonl"

    train_dataset = MusicDataset(manifest_path=train_manifest)
    valid_dataset = MusicDataset(manifest_path=valid_manifest)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    model = MusicGenModel.from_pretrained("facebook/musicgen-small")

    training_args = TrainingArguments(
        output_dir="./outputs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()

    model.save_pretrained("./trained_model")
    print("Training complete. Model saved to ./trained_model.")


def main():
    parser = argparse.ArgumentParser(description="Train MusicGen model")
    parser.add_argument("--manifest_dir", type=str, required=True, help="Path to manifest directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_model(Path(args.manifest_dir), args.epochs, args.batch_size)


if __name__ == "__main__":
    main()