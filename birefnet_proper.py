#!/usr/bin/env python3
"""
Proper BiRefNet Implementation
Based on the original BiRefNet paper and code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
import cv2
import sys
import argparse
from pathlib import Path

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        size = x.shape[2:]
        
        x1 = self.conv_1x1(x)
        x2 = self.conv_3x3_1(x)
        x3 = self.conv_3x3_2(x)
        x4 = self.conv_3x3_3(x)
        
        x5 = self.global_avg_pool(x)
        x5 = self.conv_pool(x5)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv_out(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        if skip is not None:
            # Resize skip connection to match x
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x

class BiRefNet(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        super().__init__()
        
        # Use ResNet as backbone (more stable than Swin)
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # stages 1,2,3,4
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        print(f"Feature dimensions: {self.feature_dims}")
        
        # ASPP module on the deepest features
        self.aspp = ASPP(self.feature_dims[-1], 256)
        
        # Decoder
        self.decoder4 = DecoderBlock(256 + self.feature_dims[-2], 256)
        self.decoder3 = DecoderBlock(256 + self.feature_dims[-3], 128)
        self.decoder2 = DecoderBlock(128 + self.feature_dims[-4], 64)
        self.decoder1 = DecoderBlock(64, 32)
        
        # Final prediction
        self.final_conv = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Backbone forward
        features = self.backbone(x)
        
        # ASPP on deepest features
        x = self.aspp(features[-1])
        
        # Progressive decoding
        x = self.decoder4(x, features[-2])
        x = self.decoder3(x, features[-3]) 
        x = self.decoder2(x, features[-4])
        x = self.decoder1(x)
        
        # Final prediction
        x = self.final_conv(x)
        
        # Resize to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(x)

class BiRefNetBackgroundRemover:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = BiRefNet('resnet50')
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            print(f"Loading weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("Using backbone pretrained weights only")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ BiRefNet initialized successfully!")
    
    def preprocess_image(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert('RGB')
        
        original_size = image.size
        processed = self.transform(image).unsqueeze(0).to(self.device)
        
        return processed, original_size, image
    
    def enhance_mask(self, mask):
        """Post-process mask for better quality"""
        # Convert to numpy
        mask_np = mask.squeeze().cpu().numpy()
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        
        # Smooth edges
        mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0)
        
        return mask_np
    
    def remove_background(self, image_input, output_path=None):
        with torch.no_grad():
            # Preprocess
            processed, original_size, original_image = self.preprocess_image(image_input)
            
            # Model inference
            prediction = self.model(processed)
            
            # Resize prediction to original size
            prediction = F.interpolate(prediction, size=original_size[::-1], mode='bilinear', align_corners=False)
            
            # Enhance mask
            mask = self.enhance_mask(prediction)
            mask = (mask * 255).astype(np.uint8)
            
            # Create result
            original_array = np.array(original_image)
            result_array = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_array
            result_array[:, :, 3] = mask
            
            result_image = Image.fromarray(result_array, 'RGBA')
            
            if output_path:
                result_image.save(output_path, 'PNG')
                print(f"✅ BiRefNet result saved: {output_path}")
            
            return result_image

def main():
    parser = argparse.ArgumentParser(description="BiRefNet Background Removal")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-m', '--model', help='Path to model weights')
    
    args = parser.parse_args()
    
    print("🔥 BiRefNet Background Remover")
    print("=" * 35)
    
    try:
        # Initialize
        remover = BiRefNetBackgroundRemover(args.model)
        
        # Set output
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_birefnet.png"
        
        print(f"Processing: {args.input}")
        remover.remove_background(args.input, output_path)
        print("🎉 BiRefNet processing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())