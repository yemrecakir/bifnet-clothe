#!/usr/bin/env python3
"""
Final BiRefNet Implementation with proper weights
Using transformers AutoModelForImageSegmentation
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from model_cache import get_cached_model
from PIL import Image
import numpy as np
import cv2
import sys
import argparse
from pathlib import Path

class FinalBiRefNet:
    def __init__(self, model_name='ZhengPeng7/BiRefNet', device=None, optimize_for_inference=True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        
        print("📥 Loading BiRefNet with optimizations...")
        
        # Use cached model for faster loading
        print("📦 Loading model from cache...")
        self.model = get_cached_model(
            model_name=model_name,
            device=self.device,
            optimize_for_inference=optimize_for_inference
        )
        
        print("✅ BiRefNet loaded with OPTIMIZATIONS!")
        
        # Image preprocessing parameters - smaller for speed
        self.image_size = 256  # Reduced from 320 for even faster processing
    
    def preprocess_image(self, image_input):
        """Optimized image preprocessing for BiRefNet"""
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert('RGB')
        
        original_size = image.size
        
        # Optimized resize with BILINEAR (faster than LANCZOS)
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # More efficient tensor conversion
        image_array = np.array(image_resized, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Normalize (move to device for efficiency)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=image_tensor.dtype).view(3, 1, 1)
        
        image_tensor = image_tensor.to(self.device)
        image_tensor = (image_tensor - mean) / std
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_size, image
    
    def remove_background(self, image_input, output_path=None):
        """Optimized background removal using BiRefNet"""
        # Enable autocast for mixed precision if on GPU
        autocast_context = torch.cuda.amp.autocast() if self.device.type == 'cuda' else torch.no_grad()
        
        with autocast_context:
            # Preprocess
            processed_image, original_size, original_image = self.preprocess_image(image_input)
            
            print(f"🎯 Processing image: {processed_image.shape} on {self.device}")
            
            # Optimized model inference with memory efficiency
            with torch.inference_mode():
                prediction = self.model(processed_image)
            
            # The prediction should be a tensor with segmentation mask
            if isinstance(prediction, dict):
                # Sometimes models return dict with 'logits' or 'prediction'
                if 'logits' in prediction:
                    pred_mask = prediction['logits']
                elif 'prediction' in prediction:
                    pred_mask = prediction['prediction'] 
                else:
                    pred_mask = list(prediction.values())[0]
            elif isinstance(prediction, (list, tuple)):
                pred_mask = prediction[0]  # Take first output
            else:
                pred_mask = prediction
            
            # Apply sigmoid to get probabilities
            pred_mask = torch.sigmoid(pred_mask)
            
            # Resize to original image size
            pred_mask = F.interpolate(
                pred_mask,
                size=original_size[::-1],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
            
            # Convert to numpy
            mask_np = pred_mask.squeeze().cpu().numpy()
            
            # Post-process mask
            mask_np = self.post_process_mask(mask_np)
            
            # Create transparent result
            original_array = np.array(original_image)
            result_array = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_array
            result_array[:, :, 3] = (mask_np * 255).astype(np.uint8)
            
            result_image = Image.fromarray(result_array, 'RGBA')
            
            if output_path:
                result_image.save(output_path, 'PNG')
                print(f"🔥 BiRefNet result saved: {output_path}")
            
            return result_image, Image.fromarray((mask_np * 255).astype(np.uint8), 'L')
    
    def post_process_mask(self, mask):
        """Optimized mask post-processing"""
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Reduced morphological operations for speed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Simplified processing - just close holes
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Light smoothing only
        mask_uint8 = cv2.medianBlur(mask_uint8, 3)
        
        return mask_uint8.astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser(description="Final BiRefNet Background Removal with proper weights")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-m', '--mask', help='Save mask separately')
    
    args = parser.parse_args()
    
    print("🔥🔥🔥 FINAL BiRefNet - WITH PROPER WEIGHTS 🔥🔥🔥")
    print("=" * 60)
    
    try:
        # Initialize BiRefNet
        birefnet = FinalBiRefNet()
        
        # Set output path
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_FINAL_birefnet.png"
        
        print(f"🎯 Processing: {args.input}")
        print(f"📁 Output: {output_path}")
        
        # Process image
        result, mask = birefnet.remove_background(args.input, output_path)
        
        # Save mask if requested
        if args.mask:
            mask.save(args.mask)
            print(f"🎭 Mask saved: {args.mask}")
        
        print("🎉🔥 FINAL BiRefNet processing complete! 🔥🎉")
        print("🚀 Background should be PROPERLY removed now!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())