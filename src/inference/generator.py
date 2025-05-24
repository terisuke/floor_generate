import torch
import numpy as np
from PIL import Image
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.lora_trainer import LoRATrainer

class FloorPlanGenerator:
    """平面図生成クラス - LoRAモデルを使用して敷地マスクから平面図を生成"""
    
    def __init__(self, lora_weights_path=None):
        """初期化"""
        self.trainer = LoRATrainer()
        
        if lora_weights_path and os.path.exists(lora_weights_path):
            self.trainer.load_lora_weights(lora_weights_path)
            
        from diffusers import StableDiffusionPipeline
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.trainer.model_id,
            unet=self.trainer.unet,
            torch_dtype=torch.float32 if self.trainer.device == "mps" else torch.float16
        ).to(self.trainer.device)
        
        self.pipeline.safety_checker = None
        
    def generate_plan(self, site_mask, prompt, num_inference_steps=50, guidance_scale=7.5):
        """敷地マスクとプロンプトから平面図を生成"""
        if isinstance(site_mask, np.ndarray):
            site_mask = Image.fromarray((site_mask * 255).astype(np.uint8))
        elif isinstance(site_mask, torch.Tensor):
            site_mask = Image.fromarray((site_mask.cpu().numpy()[0] * 255).astype(np.uint8))
            
        site_mask = site_mask.resize((512, 512))
        
        result = self.pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image=site_mask,
            strength=0.0  # Don't modify the image, just use it for conditioning
        )
        
        generated_image = result.images[0]
        
        generated_array = np.array(generated_image)
        
        return generated_array
        
    def save_generated_plan(self, generated_plan, output_path):
        """生成された平面図を保存"""
        if isinstance(generated_plan, np.ndarray):
            img = Image.fromarray(generated_plan)
            img.save(output_path)
            return True
        return False
