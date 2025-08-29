#!/usr/bin/env python3
"""
Download BiRefNet weights from Hugging Face
"""

import os
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForImageSegmentation

def download_birefnet_weights():
    """Download BiRefNet from Hugging Face"""
    
    print("🔥 Downloading BiRefNet from Hugging Face...")
    
    # Method 1: Download via transformers
    try:
        print("📥 Downloading BiRefNet main model...")
        model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True,
            cache_dir='./weights'
        )
        
        # Save the model locally
        os.makedirs('weights', exist_ok=True)
        torch.save(model.state_dict(), 'weights/BiRefNet.pth')
        print("✅ BiRefNet main model downloaded!")
        
        return 'weights/BiRefNet.pth'
        
    except Exception as e:
        print(f"❌ Error downloading main model: {e}")
        
    # Method 2: Download specific files
    try:
        print("📥 Trying alternative download method...")
        
        # Download specific model files
        files = [
            "pytorch_model.bin",
            "config.json",
            "model.safetensors"
        ]
        
        for file in files:
            try:
                downloaded_file = hf_hub_download(
                    repo_id="ZhengPeng7/BiRefNet",
                    filename=file,
                    cache_dir="./weights",
                    local_dir="./weights"
                )
                print(f"✅ Downloaded: {file}")
            except Exception as file_error:
                print(f"⚠️ Could not download {file}: {file_error}")
        
        return './weights/pytorch_model.bin'
        
    except Exception as e:
        print(f"❌ Error with alternative method: {e}")
    
    # Method 3: Full repo download
    try:
        print("📥 Downloading entire repository...")
        snapshot_download(
            repo_id="ZhengPeng7/BiRefNet",
            cache_dir="./weights",
            local_dir="./weights/BiRefNet_HF"
        )
        print("✅ Full repository downloaded!")
        return './weights/BiRefNet_HF'
        
    except Exception as e:
        print(f"❌ Error downloading repository: {e}")
    
    return None

if __name__ == "__main__":
    weights_path = download_birefnet_weights()
    if weights_path:
        print(f"🎉 BiRefNet weights ready at: {weights_path}")
    else:
        print("❌ Failed to download BiRefNet weights")