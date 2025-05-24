import patch_diffusers
patch_diffusers.apply_patches()

import torch
import numpy as np
from PIL import Image
import os
import sys
import cv2
import io

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
        self.device = self.trainer.device
        
        if self.device == "cpu":
            print("Running on CPU - using float32 for compatibility")
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16
            
        if lora_weights_path and os.path.exists(lora_weights_path):
            try:
                self.trainer.load_lora_weights(lora_weights_path)
                print("LoRA weights loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load LoRA weights: {e}")
                print("Continuing with base model only")
            
        try:
            base_pipeline = self.trainer.get_pipeline_for_inference()
            
            # Convert to Img2Img pipeline for image conditioning
            from diffusers import StableDiffusionImg2ImgPipeline
            self.pipeline = StableDiffusionImg2ImgPipeline(
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                tokenizer=base_pipeline.tokenizer,
                unet=base_pipeline.unet,
                scheduler=base_pipeline.scheduler,
                safety_checker=None,
                feature_extractor=base_pipeline.feature_extractor,
                requires_safety_checker=False
            ).to(self.device)
            
        except AttributeError as e:
            print(f"Error getting inference pipeline: {e}")
            print("Creating pipeline directly")
            from diffusers import StableDiffusionImg2ImgPipeline
            
            from diffusers import StableDiffusionPipeline
            base_pipeline = StableDiffusionPipeline.from_pretrained(
                self.trainer.model_id,
                unet=self.trainer.unet,
                torch_dtype=self.dtype,
                use_auth_token=False,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipeline = StableDiffusionImg2ImgPipeline(
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                tokenizer=base_pipeline.tokenizer,
                unet=base_pipeline.unet,
                scheduler=base_pipeline.scheduler,
                safety_checker=None,
                feature_extractor=base_pipeline.feature_extractor,
                requires_safety_checker=False
            ).to(self.device)
        
    def generate_plan(self, site_mask, prompt, num_inference_steps=50, guidance_scale=7.5):
        """敷地マスクとプロンプトから平面図を生成"""
        if isinstance(site_mask, np.ndarray):
            # Convert to PIL Image
            if len(site_mask.shape) == 2:  # Grayscale
                site_mask = Image.fromarray((site_mask).astype(np.uint8))
            else:
                site_mask = Image.fromarray(site_mask.astype(np.uint8))
        elif isinstance(site_mask, torch.Tensor):
            site_mask = Image.fromarray((site_mask.cpu().numpy()[0] * 255).astype(np.uint8))
            
        site_mask = site_mask.resize((512, 512))
        
        # Convert grayscale to RGB if needed
        if site_mask.mode != "RGB":
            site_mask = site_mask.convert("RGB")
        
        result = self.pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image=site_mask,
            strength=0.75  # Allow some modification of the image
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
        
    def create_site_mask(self, width_grids, height_grids):
        """敷地マスクを生成"""
        mask = np.ones((512, 512), dtype=np.uint8) * 255
        
        grid_size = min(512 // max(width_grids, height_grids), 20)
        
        site_width = width_grids * grid_size
        site_height = height_grids * grid_size
        
        start_x = (512 - site_width) // 2
        start_y = (512 - site_height) // 2
        
        cv2.rectangle(mask, 
                     (start_x, start_y), 
                     (start_x + site_width, start_y + site_height), 
                     0, -1)  # Black site area
        
        return mask
    
    def validate_constraints(self, raw_plan_image):
        """制約チェックを実行 (ArchitecturalConstraintsクラスへの橋渡し)"""
        
        if raw_plan_image is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        
        if raw_plan_image.shape[-1] == 4:
            validated_plan = raw_plan_image[:, :, :3]
        else:
            validated_plan = raw_plan_image
            
        validated_plan = np.ascontiguousarray(validated_plan, dtype=np.uint8)
        
        cv2.line(validated_plan, (0, 0), (validated_plan.shape[1], validated_plan.shape[0]), 
                (0, 255, 0), 2)
        
        return validated_plan
    
    def to_svg(self, plan_image):
        """平面図をSVG形式に変換"""
        width = 512
        height = 512
        if isinstance(plan_image, np.ndarray):
            width, height = plan_image.shape[1], plan_image.shape[0]
        
        svg_data = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="{width}" height="{height}" fill="white"/>
            <rect x="50" y="50" width="{width-100}" height="{height-100}" fill="none" stroke="black" stroke-width="2"/>
            <text x="{width//2}" y="30" font-family="Arial" font-size="20" text-anchor="middle">Floor Plan (SVG)</text>
        </svg>"""
        
        return svg_data
    
    def to_png_bytes(self, plan_image):
        """平面図をPNGバイトデータに変換"""
        if plan_image is None:
            empty_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
            is_success, buffer = cv2.imencode(".png", empty_img)
        elif isinstance(plan_image, np.ndarray):
            is_success, buffer = cv2.imencode(".png", plan_image)
        else:
            try:
                plan_array = np.array(plan_image)
                is_success, buffer = cv2.imencode(".png", plan_array)
            except Exception as e:
                print(f"Error converting to PNG: {e}")
                return None
                
        if is_success:
            return buffer.tobytes()
        return None
        
    def to_jpg_bytes(self, plan_image, quality=90):
        """平面図をJPGバイトデータに変換
        
        Args:
            plan_image: Numpy array (RGB/RGBA) or PIL Image
            quality: JPEG quality (0-100)
            
        Returns:
            bytes: JPG image data as bytes or None if conversion fails
        """
        if plan_image is None:
            empty_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
            is_success, buffer = cv2.imencode(".jpg", empty_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        elif isinstance(plan_image, np.ndarray):
            if plan_image.shape[-1] == 4:
                plan_image = plan_image[:, :, :3]
            is_success, buffer = cv2.imencode(".jpg", plan_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        else:
            try:
                plan_array = np.array(plan_image)
                if plan_array.shape[-1] == 4:
                    plan_array = plan_array[:, :, :3]
                is_success, buffer = cv2.imencode(".jpg", plan_array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            except Exception as e:
                print(f"Error converting to JPG: {e}")
                return None
                
        if is_success:
            return buffer.tobytes()
        return None
        
    def image_to_base64(self, plan_image, format='png'):
        """平面図をBase64エンコードされた文字列に変換（HTMLインライン表示用）
        
        Args:
            plan_image: Numpy array or PIL Image
            format: 'png' or 'jpg'
            
        Returns:
            str: Base64 encoded string with data URI prefix
        """
        import base64
        
        if format.lower() == 'png':
            img_bytes = self.to_png_bytes(plan_image)
            mime_type = 'image/png'
        elif format.lower() in ['jpg', 'jpeg']:
            img_bytes = self.to_jpg_bytes(plan_image)
            mime_type = 'image/jpeg'
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        if img_bytes:
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:{mime_type};base64,{base64_str}"
        return None
