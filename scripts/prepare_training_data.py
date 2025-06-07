import argparse
import os
import sys

# Add src directory to Python path to allow module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from preprocessing.training_data_generator import TrainingDataGenerator

def main():
    parser = argparse.ArgumentParser(description="Prepare training data from floor plans.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing raw floor plans.")
    parser.add_argument("--target_width", type=int, default=256,
                        help="Target width for generated images.")
    parser.add_argument("--target_height", type=int, default=256,
                        help="Target height for generated images.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"Created output directory: {args.data_dir}")

    print(f"Starting training data preparation...")
    print(f"Data directory: {args.data_dir}")
    print(f"Target image size: ({args.target_width}, {args.target_height})")

    generator = TrainingDataGenerator(target_size=(args.target_width, args.target_height))
    
    try:
        successful_count = generator.process_floor_plans(args.data_dir)
        # successful_count = generator.process_pdf_collection(args.pdf_dir, args.output_dir)
        print(f"\nTraining data preparation finished.")
        print(f"Successfully processed {successful_count} PDF files.")
        print(f"Training data pairs saved in: {args.output_dir}")
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 