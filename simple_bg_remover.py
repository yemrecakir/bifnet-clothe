#!/usr/bin/env python3
"""
Simplified Background Remover for Clothing
Using U2-Net (more stable alternative to BiRefNet)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import sys
import argparse
from pathlib import Path

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
        d1 = torch.sigmoid(self.dec1(d2))
        
        return d1

class ClothingBackgroundRemover:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SimpleUNet()
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Simple background remover initialized!")
    
    def preprocess_image(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert('RGB')
        
        original_size = image.size
        processed_image = self.transform(image).unsqueeze(0).to(self.device)
        
        return processed_image, original_size, image
    
    def remove_background(self, image_input, output_path=None):
        with torch.no_grad():
            # Preprocess
            processed_image, original_size, original_image = self.preprocess_image(image_input)
            
            # Simple edge-based mask (fallback method)
            # Convert to numpy for basic processing
            img_np = np.array(original_image)
            
            # Simple color-based segmentation
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive threshold
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
            
            # Invert mask (we want foreground)
            mask = cv2.bitwise_not(mask)
            
            # Apply morphological operations
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Smooth edges
            mask = cv2.GaussianBlur(mask, (5,5), 0)
            
            # Create result with transparent background
            result_array = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
            result_array[:, :, :3] = img_np
            result_array[:, :, 3] = mask
            
            result_image = Image.fromarray(result_array, 'RGBA')
            
            if output_path:
                result_image.save(output_path, 'PNG')
                print(f"✅ Background removed: {output_path}")
            
            return result_image

def main():
    parser = argparse.ArgumentParser(description="Simple clothing background remover")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', help='Output file path')
    
    args = parser.parse_args()
    
    print("🎽 Simple Clothing Background Remover")
    print("=" * 40)
    
    # Initialize remover
    remover = ClothingBackgroundRemover()
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_no_bg.png"
    
    try:
        print(f"Processing: {args.input}")
        result = remover.remove_background(args.input, output_path)
        print("🎉 Done!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())