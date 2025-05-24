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
    parser = argparse.ArgumentParser(description="Prepare training data from PDF floor plans.")
    parser.add_argument("--pdf_dir", type=str, default="data/raw_pdfs",
                        help="Directory containing raw PDF floor plans.")
    parser.add_argument("--output_dir", type=str, default="data/training",
                        help="Directory to save the generated training data pairs.")
    parser.add_argument("--target_width", type=int, default=256,
                        help="Target width for generated images.")
    parser.add_argument("--target_height", type=int, default=256,
                        help="Target height for generated images.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    if not os.path.exists(args.pdf_dir):
        print(f"Error: PDF directory not found: {args.pdf_dir}")
        print("Please ensure your PDF files are in the specified directory.")
        # As per user instruction, PDFs were moved to data/raw_pdfs
        # Let's create a dummy PDF in data/raw_pdfs if it's empty for testing purposes, 
        # but only if the script is run without actual PDFs.
        # This is more for robust testing than for production.
        # For now, just error out if pdf_dir is empty or doesn't exist.
        if not os.listdir(args.pdf_dir):
             print(f"PDF directory {args.pdf_dir} is empty. No data to process.")
        return

    print(f"Starting training data preparation...")
    print(f"Input PDF directory: {args.pdf_dir}")
    print(f"Output training data directory: {args.output_dir}")
    print(f"Target image size: ({args.target_width}, {args.target_height})")

    generator = TrainingDataGenerator(target_size=(args.target_width, args.target_height))
    
    try:
        successful_count = generator.process_pdf_collection(args.pdf_dir, args.output_dir)
        print(f"\nTraining data preparation finished.")
        print(f"Successfully processed {successful_count} PDF files.")
        print(f"Training data pairs saved in: {args.output_dir}")
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 