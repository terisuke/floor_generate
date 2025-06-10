import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Any

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.dataset import FloorPlanDataset
from training.lora_trainer import LoRATrainer

def custom_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """カスタムcollate関数 - Noneアイテムを処理"""
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        return {
            'condition': torch.zeros((1, 1, 512, 512), dtype=torch.float32),
            'target': torch.zeros((1, 4, 512, 512), dtype=torch.float32),
            'prompt': ['empty prompt'],
            'metadata': [{}]
        }
    
    if len(valid_batch) == 1:
        item = valid_batch[0]
        result = {}
        for key in item.keys():
            if key == 'prompt':  # Handle string prompts
                result[key] = [item[key]]
            elif key == 'metadata':  # Handle metadata dictionaries
                result[key] = [item[key]]
            else:  # Handle tensors by adding batch dimension
                result[key] = item[key].unsqueeze(0)
        return result
    
    result = {}
    for key in valid_batch[0].keys():
        if key == 'prompt':  # Handle string prompts
            result[key] = [item[key] for item in valid_batch]
        elif key == 'metadata':  # Handle metadata dictionaries
            result[key] = [item[key] for item in valid_batch]
        else:  # Handle tensors
            result[key] = torch.stack([item[key] for item in valid_batch])
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Train LoRA model for floor plan generation.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the training data pairs.")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Target size of the training data.")
    parser.add_argument("--organize_raw", action="store_true",
                        help="Organize training data from RAW directory.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1,  # Reduced to 1 for testing
                        help="Batch size for training.")
    parser.add_argument("--output_model_dir", type=str, default="models/lora_weights",
                        help="Directory to save trained LoRA model weights.")
    parser.add_argument("--load_weights_from", type=str, default=None, 
                        help="Path to pretrained LoRA weights to continue training.")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
        print(f"Error: Training data directory '{args.data_dir}' not found or is empty.")
        print("Please run `scripts/prepare_training_data.py` first.")
        return

    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir, exist_ok=True)
        print(f"Created model output directory: {args.output_model_dir}")

    print("Initializing dataset and dataloader...")
    train_dataset = FloorPlanDataset(data_dir=args.data_dir, transform=None, organize_raw=args.organize_raw, target_size=args.image_size)
    
    if len(train_dataset) == 0:
        print(f"No training data found in {args.data_dir}. Exiting.")
        return
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # num_workers=0 for MPS compatibility
        collate_fn=custom_collate_fn
    )
    print(f"Dataset loaded. Number of training samples: {len(train_dataset)}, Batches: {len(train_dataloader)}")

    print("Initializing LoRA Trainer...")
    trainer = LoRATrainer()
    
    if args.load_weights_from and os.path.exists(args.load_weights_from):
        print(f"Loading pre-trained weights from {args.load_weights_from}")
        trainer.load_lora_weights(args.load_weights_from)

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