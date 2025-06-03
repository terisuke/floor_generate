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
        glob_png = glob(f"{self.data_dir}/*.png")
        print(f"Found {len(glob_png)} json in {self.data_dir}")

        for png_path in glob_png:
            file_name = os.path.splitext(os.path.basename(png_path))[0]
            elements_json_path = f"{self.data_dir}/{file_name}_elements.json"
            integrated_json_path = f"{self.data_dir}/{file_name}_integrated.json"
            if os.path.exists(integrated_json_path):
                try:
                    with open(integrated_json_path, 'r') as f:
                        metadata = json.load(f)
                    pairs.append({
                        'metadata': metadata,
                        'png_path': png_path,
                    })
                except Exception as e:
                    print(f"Error loading metadata from {integrated_json_path}: {e}")

            else:
                 print(f"Warning: Incomplete {integrated_json_path} found. Skipping.")

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
            # Note: Applying the same random transform to both mask and plan is tricky.
            # For simple transforms like ToTensor and Normalize, this is fine.
            # For spatial transforms (rotate, crop), need to apply the same transform instance.
            # Assuming transform handles multiple inputs or is simple.
            # Let's apply transform to images before converting to tensors if it's a spatial transform,
            # or apply to tensors if it's a tensor transform.
            # For now, assume transform is simple (like normalization) applied after tensor conversion.
            # A proper implementation needs careful handling of paired transforms.
            pass # transform logic would go here if needed

        return {
            'condition': site_mask_tensor,
            'target': floor_plan_tensor,
            'prompt': prompt,
            'metadata': metadata # Optional: keep metadata for evaluation/debugging
        }

    def generate_prompt(self, metadata):
        """メタデータからプロンプト生成"""
        # Example prompt generation based on metadata with robust error handling
        try:
            grid_size = metadata.get('site_grid_size', None)
            if grid_size is None or not isinstance(grid_size, (list, tuple)) or len(grid_size) < 2:
                grid_size = ('10', '10')  # Default values if missing or invalid
                
            area = metadata.get('total_area_sqm', 0)
            if area is None:
                area = 100  # Default area
                
            rooms = metadata.get('room_count', 'N/A')
            if rooms is None:
                rooms = '4'  # Default room count
                
            source_pdf = metadata.get('source_pdf', 'unknown')
            if source_pdf is None:
                source_pdf = 'unknown'
                
            prompt = f"site_size_{grid_size[0]}x{grid_size[1]}, "
            prompt += f"total_area_{int(area) if isinstance(area, (int, float)) else area}sqm, "
            prompt += f"rooms_{rooms}, "
            
            # Clean up source name for prompt
            try:
                source_name = os.path.splitext(source_pdf)[0].replace('.','_')
            except (AttributeError, IndexError):
                source_name = 'unknown'
                
            prompt += f"source_{source_name}, "
            prompt += "japanese_house, 910mm_grid, architectural_plan"
            
            # Add more details from metadata if available and relevant for conditioning
            style = metadata.get('style', None)
            if style:
                prompt += f", style_{style}"
                
            floor_count = metadata.get('floor_count', None)
            if floor_count:
                prompt += f", floors_{floor_count}"
                
            return prompt
            
        except Exception as e:
            print(f"Error generating prompt from metadata: {e}")
            print(f"Using default prompt instead. Metadata: {metadata}")
            return "japanese_house, 910mm_grid, architectural_plan, modern_style"

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