#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import patch_diffusers
patch_diffusers.apply_patches()

from inference.generator import FloorPlanGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="Test inference with trained LoRA model")
    parser.add_argument("--weights_path", type=str, required=False, default=None,
                        help="Path to trained LoRA weights (optional)")
    parser.add_argument("--site_mask", type=str, required=True,
                        help="Path to site mask image")
    parser.add_argument("--prompt", type=str, default="japanese_house, 910mm_grid, architectural_plan",
                        help="Generation prompt")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for generated images")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of inference steps (default: 20)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.weights_path:
        print(f"Loading generator with weights from {args.weights_path}...")
        generator = FloorPlanGenerator(lora_weights_path=args.weights_path)
    else:
        print("Loading generator with base model (no trained weights)...")
        generator = FloorPlanGenerator()
    
    print(f"Loading site mask from {args.site_mask}...")
    try:
        site_mask = np.array(Image.open(args.site_mask).convert("L")) / 255.0
    except Exception as e:
        print(f"Error loading site mask: {e}")
        print("Creating a dummy site mask for testing...")
        site_mask = np.ones((512, 512), dtype=np.float32) * 0.5
    
    print(f"Generating floor plan with prompt: {args.prompt}")
    try:
        generated_plan = generator.generate_plan(
            site_mask=site_mask,
            prompt=args.prompt,
            num_inference_steps=args.steps
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        print("Exiting...")
        return
    
    output_path = os.path.join(args.output_dir, "generated_plan.png")
    generator.save_generated_plan(generated_plan, output_path)
    print(f"Generated plan saved to {output_path}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(site_mask, cmap='gray')
    axes[0].set_title("Site Mask")
    axes[0].axis('off')
    
    axes[1].imshow(generated_plan)
    axes[1].set_title("Generated Floor Plan")
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, "comparison.png")
    plt.savefig(comparison_path)
    print(f"Comparison image saved to {comparison_path}")
    
    print("Inference test completed successfully!")

if __name__ == "__main__":
    main()
