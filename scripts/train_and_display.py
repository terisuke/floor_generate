#!/usr/bin/env python
"""
End-to-end script for training and displaying floor plans.
This script integrates the full pipeline from training to visualization.
"""

import sys
import os
import subprocess
import argparse
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def run_training(epochs=20, data_dir="data/training"):
    """Run model training"""
    print("=" * 80)
    print(f"Starting model training with {epochs} epochs...")
    print("=" * 80)
    
    cmd = [sys.executable, "scripts/train_model.py", 
           "--data_dir", data_dir, "--epochs", str(epochs)]
    
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        print("\n✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Training failed with exception: {str(e)}")
        return False


def run_streamlit():
    """Launch Streamlit app"""
    print("=" * 80)
    print("Starting Streamlit app...")
    print("=" * 80)
    print("Access the app at http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    
    cmd = [sys.executable, "-m", "streamlit", "run", 
           "src/ui/main_app.py", "--server.port", "8501", 
           "--server.address", "0.0.0.0"]
    
    try:
        subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
        return True
    except KeyboardInterrupt:
        print("\n✅ Streamlit app stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Streamlit app failed with exception: {str(e)}")
        return False


def main():
    """Main function to run the end-to-end pipeline"""
    parser = argparse.ArgumentParser(description="End-to-end floor plan generation pipeline")
    parser.add_argument("--skip-training", action="store_true", 
                        help="Skip the training phase and go directly to Streamlit")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--data-dir", type=str, default="data/training",
                        help="Directory containing training data (default: data/training)")
    args = parser.parse_args()
    
    os.makedirs("outputs/svg", exist_ok=True)
    os.makedirs("outputs/freecad", exist_ok=True)
    os.makedirs("outputs/images", exist_ok=True)
    
    if not args.skip_training:
        training_success = run_training(args.epochs, args.data_dir)
        if not training_success:
            print("Warning: Training failed, but continuing to Streamlit app...")
            print("The app will use the most recent available model weights.")
            time.sleep(3)  # Give user time to read the warning
    
    run_streamlit()


if __name__ == "__main__":
    main()
