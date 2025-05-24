import torch
from torch.utils.data import Dataset
import json
import os
from glob import glob
import cv2
import numpy as np

class FloorPlanDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = self.load_data_pairs()

    def load_data_pairs(self):
        """学習データペアを読み込み"""
        pairs = []
        # Assuming data pairs are saved in subdirectories like data_dir/pair_0000, data_dir/pair_0001, etc.
        # Each subdirectory contains site_mask.png, floor_plan.png, metadata.json
        pair_dirs = glob(f"{self.data_dir}/pair_*")
        print(f"Found {len(pair_dirs)} data pairs in {self.data_dir}")

        for pair_dir in pair_dirs:
            metadata_path = f"{pair_dir}/metadata.json"
            site_mask_path = f"{pair_dir}/site_mask.png"
            floor_plan_path = f"{pair_dir}/floor_plan.png"

            if os.path.exists(metadata_path) and os.path.exists(site_mask_path) and os.path.exists(floor_plan_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    pairs.append({
                        'dir': pair_dir,
                        'metadata': metadata
                    })
                except Exception as e:
                    print(f"Error loading metadata from {metadata_path}: {e}")
            else:
                 print(f"Warning: Incomplete data pair found in {pair_dir}. Skipping.")

        print(f"Loaded {len(pairs)} valid data pairs.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if idx >= len(self.pairs):
            raise IndexError("Dataset index out of range")
            
        pair = self.pairs[idx]
        pair_dir = pair['dir']
        metadata = pair['metadata']

        # 画像読み込み
        # site_mask is grayscale, floor_plan is RGBA
        site_mask = cv2.imread(f"{pair_dir}/site_mask.png", cv2.IMREAD_GRAYSCALE)
        floor_plan = cv2.imread(f"{pair_dir}/floor_plan.png", cv2.IMREAD_UNCHANGED)

        if site_mask is None or floor_plan is None:
            # Handle error - ideally, load_data_pairs should ensure files exist
            print(f"Error loading images for pair {pair_dir}. Returning None.")
            return None # Or raise an error, or return a placeholder

        # 正規化 (0-1)
        # Ensure site_mask is HxW (no channel dim for grayscale input)
        site_mask = site_mask.astype(np.float32) / 255.0
        # Ensure floor_plan is CHW and float32
        floor_plan = floor_plan.astype(np.float32) / 255.0


        # Convert to PyTorch tensors
        # site_mask: [H, W] -> [1, H, W] for consistency with some model inputs
        site_mask_tensor = torch.from_numpy(site_mask).unsqueeze(0)
        # floor_plan: [H, W, C] -> [C, H, W] for PyTorch convention
        floor_plan_tensor = torch.from_numpy(floor_plan).permute(2, 0, 1)

        # プロンプト生成
        prompt = self.generate_prompt(metadata)

        # Apply transform if any (e.g., data augmentation)
        if self.transform:
            # Apply transforms to both site_mask_tensor and floor_plan_tensor
            transformed = self.transform(site_mask_tensor, floor_plan_tensor)
            site_mask_tensor, floor_plan_tensor = transformed

        return {
            'condition': site_mask_tensor,
            'target': floor_plan_tensor,
            'prompt': prompt,
            'metadata': metadata # Optional: keep metadata for evaluation/debugging
        }

    def generate_prompt(self, metadata):
        """メタデータからプロンプト生成"""
        # Enhanced prompt generation based on metadata
        grid_size = metadata.get('site_grid_size', ('N/A', 'N/A'))
        area = metadata.get('total_area_sqm', 0)
        rooms = metadata.get('room_count', 'N/A')
        source_pdf = metadata.get('source_pdf', 'unknown')
        
        prompt = f"site_size_{grid_size[0]}x{grid_size[1]}, "
        if area is not None:
            prompt += f"total_area_{area:.0f}sqm, "
        prompt += f"rooms_{rooms}, "
        prompt += f"source_{os.path.splitext(source_pdf)[0].replace('.','_')}, " # Clean up source name for prompt
        
        room_types = metadata.get('room_types', [])
        if room_types:
            prompt += f"rooms_{'_'.join(room_types)}, "
            
        floor_count = metadata.get('floor_count', None)
        if floor_count:
            prompt += f"{floor_count}_floor, "
            
        style = metadata.get('style', None)
        if style:
            prompt += f"{style}_style, "
            
        prompt += "japanese_house, 910mm_grid, architectural_plan"

        return prompt

# Example of a simple transform (optional)
# from torchvision import transforms
# class ToTensorAndNormalize:
#      def __call__(self, image, mask):
#          # Convert numpy arrays to tensors
#          image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
#          mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
#          # Apply normalization if needed
#          # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Example for 3 channel image
#          # image_tensor = normalize(image_tensor)
#          return image_tensor, mask_tensor  