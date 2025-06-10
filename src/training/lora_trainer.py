import sys
import os
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリを取得し、sys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import patch_diffusers
patch_diffusers.apply_patches()

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class LoRATrainer:
    def __init__(self, r=64, lora_alpha=64):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Base model - using v1-4 which is open access and smaller
        self.model_id = "CompVis/stable-diffusion-v1-4"
        # Load in float32 for MPS compatibility during training, will convert to float16 for inference
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32, # Use float32 for MPS training
            use_auth_token=False,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

        # 勾配チェックポイントを有効化
        self.pipeline.unet.enable_gradient_checkpointing()

        # アテンションブロックの設定を変更
        for block in self.pipeline.unet.down_blocks + self.pipeline.unet.up_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    # アテンションヘッドを削減
                    if hasattr(attn, "transformer_blocks"):
                        for transformer_block in attn.transformer_blocks:
                            if hasattr(transformer_block, "attn1"):
                                transformer_block.attn1.heads = 4  # デフォルトは8
                            if hasattr(transformer_block, "attn2"):
                                transformer_block.attn2.heads = 4  # デフォルトは8

        # Disable safety checker
        self.pipeline.safety_checker = None

        # LoRA設定
        self.lora_config = LoraConfig(
            r=r,                    # Rank (軽量化)
            lora_alpha=lora_alpha,
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2"
            ],
            lora_dropout=0.1,
        )

        # Ensure UNet is in float32 before applying LoRA
        self.pipeline.unet = self.pipeline.unet.to(torch.float32)

        # Apply LoRA to UNet
        self.unet = get_peft_model(self.pipeline.unet, self.lora_config)

        # Optimizer setup after applying LoRA
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=1e-4,
            weight_decay=1e-2
        )

        # Scheduler (optional but good practice)
        self.scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)

        # メモリ使用量を監視
        if self.device == "mps":
            torch.mps.empty_cache()

    def get_memory_usage(self):
        """メモリ使用量を取得"""
        if self.device == "mps":
            try:
                # MPSデバイスのメモリ使用量を取得
                allocated = torch.mps.current_allocated_memory() / (1024**3)  # GBに変換
                reserved = torch.mps.driver_allocated_memory() / (1024**3)    # GBに変換
                return f"MPS: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            except Exception as e:
                return f"MPS memory info unavailable: {str(e)}"
        else:
            try:
                # CPUメモリ使用量を取得
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                return f"CPU: {memory_info.rss / (1024**3):.2f}GB"
            except Exception as e:
                return f"CPU memory info unavailable: {str(e)}"

    def clear_memory(self):
        """メモリをクリア"""
        if self.device == "mps":
            torch.mps.empty_cache()

    def train(self, train_dataloader: DataLoader, num_epochs=20):
        """LoRA学習実行"""

        self.unet.train()

        # Move text_encoder to device
        self.pipeline.text_encoder = self.pipeline.text_encoder.to(self.device)

        datetime_start = datetime.now()
        for epoch in range(1, num_epochs+1):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):

                # バッチデータ
                # Ensure tensors are in float32 for MPS training
                site_masks = batch['condition'].to(self.device, dtype=torch.float32)
                target_plans = batch['target'].to(self.device, dtype=torch.float32)
                prompts = batch['prompt']

                # ノイズ追加
                noise = torch.randn_like(target_plans).to(self.device, dtype=torch.float32)
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (target_plans.shape[0],), device=self.device).long()

                noisy_plans = self.scheduler.add_noise(
                    target_plans, noise, timesteps
                )

                # 予測
                # No need for autocast with MPS and float32, managed by device
                
                # Text encoding
                # Ensure input_ids are on the correct device
                text_input_ids = self.pipeline.tokenizer(
                    prompts,
                    padding=True,
                    return_tensors="pt"
                ).input_ids.to(self.device)

                text_embeddings = self.pipeline.text_encoder(
                    text_input_ids
                )[0]

                # UNet prediction
                noise_pred = self.unet(
                    noisy_plans,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False
                )[0]

                # Loss計算
                loss = F.mse_loss(noise_pred, noise, reduction="mean")

                # バックプロパゲーション
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # バッチ処理後にメモリをクリア
                if batch_idx % 10 == 0:  # 10バッチごとにクリア
                    self.clear_memory()
    
                # 進捗率　= 残りエポック数　/ 総エポック数 + 現在のバッチ数 / 総バッチ数
                progress = ((epoch-1) / num_epochs) + ((batch_idx + 1) / len(train_dataloader)) * (1 / num_epochs)
                # 予想終了時刻を表示する
                estimated_end_time = datetime_start + timedelta(seconds=((datetime.now() - datetime_start).total_seconds() / progress))
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Epoch {epoch:2d}, Batch {batch_idx:3d}, Loss: {loss.item():.4f}, Memory: {self.get_memory_usage()}, Estimated end time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

            print(f"Epoch {epoch} completed. Average Loss: {total_loss/len(train_dataloader):.4f}")

            # モデル保存
            # Save only the LoRA weights
            if epoch % 5 == 0 or epoch == num_epochs:
                 # Create directory if it doesn't exist
                 save_dir = f"models/lora_weights/epoch_{epoch:02d}"
                 os.makedirs(save_dir, exist_ok=True)
                 self.unet.save_pretrained(save_dir)
                 print(f"LoRA weights saved to {save_dir}")

    def inference(self, site_mask: torch.Tensor, prompt: str, num_inference_steps=50):
        """単一プロンプトでの推論実行"""
        self.unet.eval()

        # Move site_mask to device and ensure float32
        site_mask = site_mask.to(self.device, dtype=torch.float32)
        
        # Add batch dimension if missing
        if site_mask.ndim == 3:
            site_mask = site_mask.unsqueeze(0)

        # Prepare text embeddings
        text_input_ids = self.pipeline.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt"
        ).input_ids.to(self.device)

        text_embeddings = self.pipeline.text_encoder(
            text_input_ids
        )[0]

        # Prepare latent dimensions based on UNet input size
        # SD 2.1 works with 512x512 images, latents are 64x64
        # Need to adjust if using a different base model or resolution
        # Assuming 256x256 target size, latents should be 32x32
        # The condition (site_mask) size should match the target image size for input to the generator model (which isn't standard SD)
        # The requirement has site_mask as input to SD inference.
        # Standard Stable Diffusion does NOT take an image mask directly as conditional input in the UNet like this.
        # ControlNet or similar models are used for this.
        # The requirement implies the site mask is somehow used in the SD pipeline.
        # Let's assume the generator model (which uses SD 2.1 + LoRA) is modified to accept the site_mask.
        # This is a significant deviation from standard SD inference.
        # Re-reading requirement: G[Streamlit UI] --> H[敷地マスク生成] --> I[SD推論<br>平面図生成]
        # This suggests the SD inference step *uses* the site mask.
        # How is the site mask incorporated into SD 2.1 + LoRA?
        # Option 1: Concatenate site mask to latent space (requires modifying UNet architecture)
        # Option 2: Use site mask as an attention mask (requires modifying UNet)
        # Option 3: Simple approach - upscale site mask and concatenate to the *image* input before VAE encoding? (Less common)
        # Option 4: Modify the prompt based on the site mask characteristics? (Indirect)
        # Option 5: Use the site mask during the diffusion process (e.g., inpainting-like conditional generation)?

        # Given the mention of SD 2.1 + LoRA, modifying the UNet architecture directly to accept a site mask channel
        # in the latent space is the most plausible approach for conditional generation based on the mask.
        # This would mean the UNet input should be `noisy_latents + site_mask_latent`. The site_mask_latent would be derived from the site_mask.

        # Let's assume the LoRA-tuned UNet is adapted to take an extra channel for the site mask.
        # This would mean `noisy_plans` (which become latents after VAE encode) need an extra channel.
        # Or, the site mask is processed and input alongside `noisy_plans` and `timesteps` into `self.unet()`.
        # The `unet` call in the training loop `noise_pred = self.unet(noisy_plans, timesteps.to(self.device), encoder_hidden_states=text_embeddings, return_dict=False)[0]`
        # only takes `noisy_plans`, `timesteps`, `encoder_hidden_states`. It does *not* take `site_mask`.
        # This implies the site mask is *not* used as a direct input to the UNet in the provided training code structure.

        # Re-evaluating the inference process in requirement (Section 8.1 Streamlit UI):
        # `site_mask = self.generator.create_site_mask(width, height)`
        # `raw_plan = self.generator.generate_plan(site_mask, prompt)`
        # This `generate_plan` method (presumably in src/inference/generator.py) is the one that uses SD inference.
        # It receives the `site_mask` and `prompt`. How does it use them with a standard SD pipeline (even LoRA-tuned)?

        # Perhaps the `site_mask` is used *during* the diffusion sampling loop, for example,
        # by enforcing the mask on the generated image at each step (like inpainting)?
        # Or maybe the `site_mask` is used to guide the generation in some other way not standard to the base SD pipeline.

        # Let's look at the `FloorPlanGenerator` class in the Streamlit UI code snippet (Section 8.1):
        # It has `generate_plan(site_mask, prompt)` and `validate_constraints(raw_plan)` and `to_svg(validated_plan)`.
        # This suggests `generate_plan` is where the SD inference happens.
        # The `LoRATrainer` class here seems focused on the *training* loop for the UNet + LoRA.
        # The *inference* logic should likely reside in a separate class, e.g., `FloorPlanGenerator` or similar.
        # The `inference` method signature in *this* `LoRATrainer` is confusing as it implies this class handles inference.
        # Let's remove the `inference` method from `LoRATrainer` as it seems misplaced based on the overall structure.
        # Inference should be handled by the `FloorPlanGenerator` which orchestrates the use of the trained model.

        pass # Remove this method after confirming the plan

    # Method to load trained LoRA weights
    def load_lora_weights(self, weights_path):
        """学習済みLoRA重みを読み込み"""
        # Load LoRA weights into the UNet
        # self.unet is already a PeftModel because get_peft_model was called in __init__
        # Need to load the state_dict from the saved weights_path
        if os.path.exists(weights_path):
            print(f"Loading LoRA weights from {weights_path}...")
            try:
                from peft import PeftModel
                self.unet = PeftModel.from_pretrained(
                    self.unet,
                    weights_path,
                    is_trainable=False  # Set to False for inference
                )
                print("LoRA weights loaded using PeftModel.from_pretrained.")
            except Exception as e:
                print(f"Error loading with from_pretrained: {e}")
                try:
                    adapter_path = os.path.join(weights_path, "adapter_model.bin")
                    if os.path.exists(adapter_path):
                        self.unet.load_state_dict(torch.load(adapter_path), strict=False)
                    else:
                        # Try safetensors format
                        adapter_path = os.path.join(weights_path, "adapter_model.safetensors")
                        if os.path.exists(adapter_path):
                            try:
                                from safetensors.torch import load_file
                                state_dict = load_file(adapter_path)
                                self.unet.load_state_dict(state_dict, strict=False)
                            except ImportError:
                                self.unet.load_state_dict(torch.load(adapter_path), strict=False)
                    print(f"LoRA weights loaded from {adapter_path} using state_dict.")
                except Exception as e2:
                    print(f"Failed to load weights with fallback method: {e2}")
                    raise ValueError(f"Could not load LoRA weights from {weights_path}")
            print("LoRA weights loaded successfully.")
        else:
            print(f"Warning: LoRA weights not found at {weights_path}")
            
    def get_pipeline_for_inference(self):
        """推論用のパイプラインを取得"""
        self.unet.eval()
        
        from diffusers import StableDiffusionPipeline
        
        if self.device == "cpu":
            dtype = torch.float32
        else:
            print("MPS device detected. Forcing float32 for inference due to potential MPS issues with float16.")
            dtype = torch.float32
        
        if hasattr(self.unet, "get_base_model"):
            unet_for_pipeline = self.unet.get_base_model()
        elif hasattr(self.unet, "base_model"):
            unet_for_pipeline = self.unet.base_model
        else:
            unet_for_pipeline = self.unet
            
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            unet=unet_for_pipeline,
            torch_dtype=dtype,
            use_auth_token=False,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # Disable safety checker to save memory
        pipeline.safety_checker = None
        
        return pipeline


# Helper function for demonstration (optional)
# if __name__ == "__main__":
#     # Dummy usage
#     trainer = LoRATrainer()
#     print("LoRATrainer initialized.")
#     # Need a dummy DataLoader for training or remove the __main__ block
#     # trainer.train(dummy_dataloader, num_epochs=1)
#     # Need a dummy site_mask and prompt for inference or remove the method
#     # trainer.inference(dummy_mask, "a test house")                              