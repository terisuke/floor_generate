#!/usr/bin/env python3
"""
Floor Plan Generation Inference Test Script
This script tests the inference pipeline using a trained LoRA model.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inference.generator import FloorPlanGenerator

def main():
    parser = argparse.ArgumentParser(description="Test floor plan generation inference.")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to trained LoRA weights.")
    parser.add_argument("--site_mask", type=str, required=True,
                        help="Path to site mask image.")
    parser.add_argument("--prompt", type=str, 
                        default="japanese_house, 910mm_grid, architectural_plan",
                        help="Prompt for generation.")
    parser.add_argument("--output_dir", type=str, default="outputs/generated",
                        help="Directory to save generated floor plans.")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights_path):
        print(f"Error: LoRA weights not found at {args.weights_path}")
        return
        
    if not os.path.exists(args.site_mask):
        print(f"Error: Site mask not found at {args.site_mask}")
        return
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    site_mask = Image.open(args.site_mask).convert("L")
    site_mask_array = np.array(site_mask) / 255.0
    
    print("Initializing FloorPlanGenerator...")
    generator = FloorPlanGenerator(lora_weights_path=args.weights_path)
    
    print(f"Generating floor plan with prompt: {args.prompt}")
    print(f"Using {args.steps} inference steps")
    generated_plan = generator.generate_plan(
        site_mask=site_mask_array,
        prompt=args.prompt,
        num_inference_steps=args.steps
    )
    
    output_path = os.path.join(args.output_dir, "generated_plan.png")
    generator.save_generated_plan(generated_plan, output_path)
    print(f"Generated floor plan saved to {output_path}")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Site Mask")
    plt.imshow(site_mask_array, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Floor Plan")
    plt.imshow(generated_plan)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "comparison.png"))
    print(f"Comparison image saved to {os.path.join(args.output_dir, 'comparison.png')}")
    
if __name__ == "__main__":
    main()
