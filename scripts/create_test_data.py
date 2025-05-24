#!/usr/bin/env python3

import os
import sys
import numpy as np
from PIL import Image
import json

def create_test_site_mask(output_dir, width=512, height=512):
    """Create a simple test site mask for inference testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    site_mask = np.ones((height, width), dtype=np.uint8) * 128
    
    border_width = 20
    site_mask[:border_width, :] = 200
    site_mask[-border_width:, :] = 200
    site_mask[:, :border_width] = 200
    site_mask[:, -border_width:] = 200
    
    center_x, center_y = width // 2, height // 2
    building_width, building_height = width // 3, height // 3
    
    x1 = center_x - building_width // 2
    y1 = center_y - building_height // 2
    x2 = center_x + building_width // 2
    y2 = center_y + building_height // 2
    
    site_mask[y1:y2, x1:x2] = 50
    
    img = Image.fromarray(site_mask)
    site_mask_path = os.path.join(output_dir, "site_mask.png")
    img.save(site_mask_path)
    
    floor_plan = np.zeros((height, width, 4), dtype=np.uint8)
    floor_plan[:, :, 0] = 255  # Red channel
    floor_plan[:, :, 3] = 255  # Alpha channel
    
    floor_plan[y1:y2, x1:x2, 0] = 0
    floor_plan[y1:y2, x1:x2, 1] = 0
    floor_plan[y1:y2, x1:x2, 2] = 0
    floor_plan[y1:y2, x1:x2, 3] = 255
    
    floor_plan_img = Image.fromarray(floor_plan)
    floor_plan_path = os.path.join(output_dir, "floor_plan.png")
    floor_plan_img.save(floor_plan_path)
    
    metadata = {
        "site_grid_size": [10, 10],
        "total_area_sqm": 100,
        "room_count": 4,
        "source_pdf": "test.pdf",
        "room_types": ["living", "kitchen", "bedroom", "bathroom"],
        "floor_count": 1,
        "style": "modern"
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created test data in {output_dir}")
    print(f"Site mask: {site_mask_path}")
    print(f"Floor plan: {floor_plan_path}")
    print(f"Metadata: {metadata_path}")
    
    return site_mask_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create test data for inference testing")
    parser.add_argument("--output_dir", type=str, default="data/training/pair_0001",
                        help="Output directory for test data")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of the test images")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of the test images")
    
    args = parser.parse_args()
    
    create_test_site_mask(args.output_dir, args.width, args.height)
