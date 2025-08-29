#!/usr/bin/env python3
"""
Real BiRefNet Implementation 
Using the official BiRefNet model from ZhengPeng7/BiRefNet
"""

import sys
import os

# Add BiRefNet to path
sys.path.insert(0, '/Users/emrebirinci/Desktop/bff/BiRefNet')

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import argparse
from pathlib import Path

# Import BiRefNet modules
from models.birefnet import BiRefNet
from config import Config
from utils import save_tensor_img

class RealBiRefNetRemover:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        
        # Initialize config
        self.config = Config()
        
        # Load model
        print("Loading BiRefNet model...")
        self.model = BiRefNet(bb_pretrained=True)
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            print(f"Loading weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print("✅ Custom weights loaded!")
        else:
            print("⚠️ Using backbone pretrained weights only (for better results, download BiRefNet weights)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        print("✅ Real BiRefNet initialized!")
    
    def preprocess_image(self, image_input, target_size=1024):
        """Preprocess image for BiRefNet"""
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert('RGB')
        
        original_size = image.size
        
        # Resize to target size while maintaining aspect ratio
        image = self.resize_image(image, target_size)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        for i in range(3):
            image_tensor[i] = (image_tensor[i] - self.mean[i]) / self.std[i]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_size, image
    
    def resize_image(self, image, target_size):
        """Resize image while maintaining aspect ratio"""
        w, h = image.size
        
        # Calculate new dimensions
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        # Make dimensions divisible by 32 (important for BiRefNet)
        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32
        
        return image.resize((new_w, new_h), Image.LANCZOS)
    
    def remove_background(self, image_input, output_path=None, target_size=1024):
        """Remove background using real BiRefNet"""
        with torch.no_grad():
            # Preprocess
            processed_image, original_size, resized_image = self.preprocess_image(image_input, target_size)
            
            print(f"Processing image: {processed_image.shape}")
            
            # Model inference
            predictions = self.model(processed_image)
            
            # Get the final prediction (last in the list)
            if isinstance(predictions, list):
                final_pred = predictions[-1]
            else:
                final_pred = predictions
            
            # Apply sigmoid to get probabilities
            mask = torch.sigmoid(final_pred)
            
            # Resize mask back to original image size
            mask_resized = F.interpolate(
                mask, 
                size=original_size[::-1],  # PIL size is (w,h), torch expects (h,w)
                mode='bilinear', 
                align_corners=False
            )
            
            # Convert mask to numpy
            mask_np = mask_resized.squeeze().cpu().numpy()
            
            # Post-process mask
            mask_np = self.post_process_mask(mask_np)
            
            # Load original image for final composition
            if isinstance(image_input, str):
                original_image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                original_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                original_image = image_input.convert('RGB')
            
            # Create result with transparent background
            original_array = np.array(original_image)
            result_array = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_array
            result_array[:, :, 3] = (mask_np * 255).astype(np.uint8)
            
            result_image = Image.fromarray(result_array, 'RGBA')
            
            # Save result
            if output_path:
                result_image.save(output_path, 'PNG')
                print(f"🔥 Real BiRefNet result saved: {output_path}")
            
            return result_image
    
    def post_process_mask(self, mask):
        """Post-process mask for better quality"""
        # Apply morphological operations
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Close small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_small)
        
        # Gaussian blur for smooth edges
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (3, 3), 0)
        
        return mask_uint8.astype(np.float32) / 255.0

def download_weights():
    """Download BiRefNet weights"""
    print("📥 For best results, download BiRefNet weights from:")
    print("🔗 https://github.com/ZhengPeng7/BiRefNet")
    print("📁 Place weights in 'weights/' folder")
    print("🎯 Recommended: BiRefNet-general-epoch_244.pth")

def main():
    parser = argparse.ArgumentParser(description="Real BiRefNet Background Removal")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-w', '--weights', help='Path to BiRefNet weights')
    parser.add_argument('-s', '--size', type=int, default=1024, help='Target image size (default: 1024)')
    
    args = parser.parse_args()
    
    print("🔥🔥🔥 REAL BiRefNet Background Remover 🔥🔥🔥")
    print("=" * 50)
    
    try:
        # Check if weights exist
        weights_path = args.weights
        if not weights_path:
            # Look for weights in common locations
            possible_paths = [
                'weights/BiRefNet-general-epoch_244.pth',
                '../weights/BiRefNet-general-epoch_244.pth',
                'BiRefNet-general-epoch_244.pth'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    weights_path = path
                    break
        
        if not weights_path or not os.path.exists(weights_path):
            print("⚠️ No weights found, using backbone pretrained weights only")
            download_weights()
            weights_path = None
        
        # Initialize remover
        remover = RealBiRefNetRemover(weights_path)
        
        # Set output path
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_real_birefnet.png"
        
        print(f"🎯 Processing: {args.input}")
        print(f"📁 Output: {output_path}")
        
        # Process image
        result = remover.remove_background(args.input, output_path, target_size=args.size)
        
        print("🎉🔥 REAL BiRefNet processing complete! 🔥🎉")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())