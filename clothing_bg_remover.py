#!/usr/bin/env python3
"""
Clothing Background Remover using BiRefNet
Specialized for clothing and fashion item background removal
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import cv2

from birefnet_model import BiRefNetBackgroundRemover

class ClothingBackgroundRemover:
    def __init__(self, model_path=None, device=None):
        """Initialize the clothing-specific background remover"""
        self.remover = BiRefNetBackgroundRemover(model_path, device)
        
        # Clothing-specific post-processing parameters
        self.clothing_categories = [
            'shirt', 'pants', 'dress', 'jacket', 'shoes', 
            'hat', 'bag', 'accessory', 'top', 'bottom'
        ]
        
    def enhance_clothing_mask(self, mask, image):
        """
        Apply clothing-specific mask enhancements
        """
        # Convert to numpy if needed
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small holes
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Apply edge smoothing
        mask_smoothed = cv2.GaussianBlur(mask_cleaned, (3, 3), 0)
        
        return mask_smoothed
    
    def remove_background(self, image_input, output_path=None, enhance_mask=True):
        """
        Remove background from clothing image with enhancements
        
        Args:
            image_input: Input image (path, PIL Image, or numpy array)
            output_path: Output file path (optional)
            enhance_mask: Whether to apply clothing-specific enhancements
            
        Returns:
            PIL Image with transparent background
        """
        # Get initial result
        result, mask = self.remover.remove_background(image_input, return_mask=True)
        
        if enhance_mask:
            # Apply clothing-specific enhancements
            enhanced_mask = self.enhance_clothing_mask(mask, result)
            
            # Recreate result with enhanced mask
            if isinstance(image_input, str):
                original_image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                original_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                original_image = image_input.convert('RGB')
            
            original_array = np.array(original_image)
            result_array = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_array
            result_array[:, :, 3] = enhanced_mask
            
            result = Image.fromarray(result_array, 'RGBA')
        
        # Save result
        if output_path:
            result.save(output_path, 'PNG')
            print(f"✓ Clothing background removed: {output_path}")
        
        return result
    
    def process_clothing_folder(self, input_folder, output_folder):
        """
        Process all images in a folder
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process...")
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing [{i}/{len(image_files)}]: {image_file.name}")
            
            try:
                output_file = output_path / f"{image_file.stem}_no_bg.png"
                self.remove_background(str(image_file), str(output_file))
                
            except Exception as e:
                print(f"✗ Error processing {image_file.name}: {e}")
        
        print(f"✓ Processing complete! Results saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Remove background from clothing images using BiRefNet")
    parser.add_argument('input', help='Input image file or folder path')
    parser.add_argument('-o', '--output', help='Output file or folder path')
    parser.add_argument('-m', '--model', help='Path to BiRefNet model weights')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--no-enhance', action='store_true', help='Disable clothing-specific enhancements')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🎽 Clothing Background Remover - BiRefNet Edition")
    print("=" * 50)
    print(f"Device: {device}")
    
    # Initialize remover
    try:
        remover = ClothingBackgroundRemover(args.model, device)
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return 1
    
    # Check if input is file or folder
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file processing
        if args.output:
            output_file = args.output
        else:
            output_file = input_path.parent / f"{input_path.stem}_no_bg.png"
        
        print(f"Processing: {input_path}")
        try:
            remover.remove_background(
                str(input_path), 
                str(output_file), 
                enhance_mask=not args.no_enhance
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
            
    elif input_path.is_dir():
        # Folder processing
        if args.output:
            output_folder = args.output
        else:
            output_folder = input_path.parent / f"{input_path.name}_no_bg"
        
        try:
            remover.process_clothing_folder(str(input_path), str(output_folder))
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    else:
        print(f"❌ Input path does not exist: {input_path}")
        return 1
    
    print("🎉 Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())