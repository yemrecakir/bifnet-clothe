#!/usr/bin/env python3
"""
Download BiRefNet model during Docker build
"""
import os
import sys

try:
    # Set cache directories
    os.environ['TRANSFORMERS_CACHE'] = '/app/models'
    os.environ['HF_HOME'] = '/app/models'
    os.makedirs('/app/models', exist_ok=True)
    
    print("📥 Pre-downloading BiRefNet model...")
    
    from transformers import AutoModelForImageSegmentation
    
    model = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet',
        trust_remote_code=True,
        cache_dir='/app/models'
    )
    
    print("✅ BiRefNet model downloaded and cached successfully!")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("⚠️ Model will be downloaded on first API request instead")
    # Don't fail the build, just continue without pre-cached model
    sys.exit(0)