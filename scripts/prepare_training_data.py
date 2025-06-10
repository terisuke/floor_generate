import argparse
import os
import sys

# Add src directory to Python path to allow module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.dataset import FloorPlanDataset

def main():
    parser = argparse.ArgumentParser(description="Prepare training data from floor plans.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the training data pairs.")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Target size of the training data.")
    parser.add_argument("--organize_raw", action="store_true",
                        help="Organize training data from RAW directory.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"Created output directory: {args.data_dir}")

    print(f"Starting training data preparation...")
    print(f"Data directory: {args.data_dir}")
    print(f"Target image size: ({args.image_size})")

    try:
        train_dataset = FloorPlanDataset(data_dir=args.data_dir, transform=None, organize_raw=args.organize_raw, target_size=args.image_size)
        print(f"\nTraining data preparation finished.")
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 