import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.dataset import FloorPlanDataset
from training.lora_trainer import LoRATrainer

def main():
    parser = argparse.ArgumentParser(description="Train LoRA model for floor plan generation.")
    parser.add_argument("--data_dir", type=str, default="data/training",
                        help="Directory containing the training data pairs.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, # Adjusted for potentially larger data
                        help="Batch size for training.")
    parser.add_argument("--output_model_dir", type=str, default="models/lora_weights",
                        help="Directory to save trained LoRA model weights.")
    # Add argument for loading pre-trained weights if needed for fine-tuning/resume
    # parser.add_argument("--load_weights_from", type=str, default=None, help="Path to pretrained LoRA weights to continue training.")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
        print(f"Error: Training data directory '{args.data_dir}' not found or is empty.")
        print("Please run `scripts/prepare_training_data.py` first.")
        return

    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir, exist_ok=True)
        print(f"Created model output directory: {args.output_model_dir}")

    print("Initializing dataset and dataloader...")
    # Add any transforms if necessary, e.g., from torchvision
    # transform = ... 
    train_dataset = FloorPlanDataset(data_dir=args.data_dir, transform=None)
    
    if len(train_dataset) == 0:
        print(f"No training data found in {args.data_dir}. Exiting.")
        return
        
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers=0 for MPS compatibility if issues arise
    print(f"Dataset loaded. Number of training samples: {len(train_dataset)}, Batches: {len(train_dataloader)}")

    print("Initializing LoRA Trainer...")
    trainer = LoRATrainer()
    # if args.load_weights_from:
    #    trainer.load_lora_weights(args.load_weights_from) # Implement this in LoRATrainer

    print(f"Starting training for {args.epochs} epochs...")
    try:
        trainer.train(train_dataloader, num_epochs=args.epochs)
        print("Training finished.")
        print(f"Trained LoRA model weights saved in: {args.output_model_dir}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 