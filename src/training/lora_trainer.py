from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

class LoRATrainer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Base model
        self.model_id = "runwayml/stable-diffusion-v2-1"
        # Load in float32 for MPS compatibility during training, will convert to float16 for inference
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32 # Use float32 for MPS training
        ).to(self.device)

        # Disable safety checker
        self.pipeline.safety_checker = None

        # LoRA設定
        self.lora_config = LoraConfig(
            r=64,                    # Rank (軽量化)
            lora_alpha=64,
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


    def train(self, train_dataloader: DataLoader, num_epochs=20):
        """LoRA学習実行"""

        self.unet.train()

        # Move text_encoder to device
        self.pipeline.text_encoder = self.pipeline.text_encoder.to(self.device)

        for epoch in range(num_epochs):
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

                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            print(f"Epoch {epoch} completed. Average Loss: {total_loss/len(train_dataloader):.4f}")

            # モデル保存
            # Save only the LoRA weights
            if epoch % 5 == 0:
                 # Create directory if it doesn't exist
                 save_dir = f"models/lora_weights/epoch_{epoch}"
                 os.makedirs(save_dir, exist_ok=True)
                 self.unet.save_pretrained(save_dir)
                 print(f"LoRA weights saved to {save_dir}")

    def get_pipeline_for_inference(self):
        """推論用のパイプラインを取得"""
        # This is a helper method to get a pipeline with the trained LoRA weights
        
        inference_pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            unet=self.unet,  # Use the LoRA-trained UNet
            torch_dtype=torch.float32 if self.device == "mps" else torch.float16
        ).to(self.device)
        
        # Disable safety checker for architectural floor plans
        inference_pipeline.safety_checker = None
        
        return inference_pipeline

    # Method to load trained LoRA weights
    def load_lora_weights(self, weights_path):
        """学習済みLoRA重みを読み込み"""
        # Load LoRA weights into the UNet
        # self.unet is already a PeftModel because get_peft_model was called in __init__
        # Need to load the state_dict from the saved weights_path
        if os.path.exists(weights_path):
            print(f"Loading LoRA weights from {weights_path}...")
            self.unet.load_state_dict(torch.load(os.path.join(weights_path, "adapter_model.safetensors")), strict=False) # Assuming safetensors format
            # Or if saved with save_pretrained, it might be different.
            # Let's assume save_pretrained saves the correct format for load_pretrained.
            self.unet = self.unet.from_pretrained(self.unet, weights_path) # Correct way to load with peft

            print("LoRA weights loaded.")
        else:
            print(f"Warning: LoRA weights not found at {weights_path}")


# Helper function for demonstration (optional)
# if __name__ == "__main__":
#     # Dummy usage
#     trainer = LoRATrainer()
#     print("LoRATrainer initialized.")
#     # Need a dummy DataLoader for training or remove the __main__ block
#     # trainer.train(dummy_dataloader, num_epochs=1)
#     # Need a dummy site_mask and prompt for inference or remove the method
#     # trainer.inference(dummy_mask, "a test house")  