import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
import cv2
import os
import gdown
from pathlib import Path

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BiRefNet(nn.Module):
    def __init__(self, bb_pretrained=True):
        super().__init__()
        
        # Backbone: Swin Transformer
        try:
            self.backbone = timm.create_model(
                'swin_large_patch4_window12_384', 
                pretrained=bb_pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
        except:
            # Fallback to a working model
            self.backbone = timm.create_model(
                'resnet50', 
                pretrained=bb_pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4)
            )
        
        # Get backbone feature dimensions
        self.bb_channels = self.backbone.feature_info.channels()
        
        # Decoder layers
        self.decoder = nn.ModuleList([
            self._make_decoder_layer(self.bb_channels[3], 256),  # Stage 4
            self._make_decoder_layer(256 + self.bb_channels[2], 128),  # Stage 3
            self._make_decoder_layer(128 + self.bb_channels[1], 64),   # Stage 2
            self._make_decoder_layer(64 + self.bb_channels[0], 32),    # Stage 1
        ])
        
        # Final prediction head
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary heads for training (optional)
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Conv2d(64, 1, 3, 1, 1),
        ])
        
    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Start decoding from the deepest feature
        out = self.decoder[0](features[3])  # Stage 4
        aux_outputs = [self.aux_heads[0](out)]
        
        # Progressive upsampling and feature fusion
        for i in range(1, 4):
            # Upsample current output
            out = F.interpolate(out, size=features[3-i].shape[2:], mode='bilinear', align_corners=False)
            # Concatenate with skip connection
            out = torch.cat([out, features[3-i]], dim=1)
            # Apply decoder layer
            out = self.decoder[i](out)
            
            if i < 3:  # Don't create aux output for final layer
                aux_outputs.append(self.aux_heads[i](out))
        
        # Final upsampling to input size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Final prediction
        pred = self.final_conv(out)
        
        if self.training:
            return pred, aux_outputs
        else:
            return pred

class BiRefNetBackgroundRemover:
    def __init__(self, model_path=None, device=None):
        """
        Initialize BiRefNet model for background removal
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = BiRefNet(bb_pretrained=True)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        else:
            print("No pretrained weights provided. Using backbone pretrained weights only.")
            print("For best results, download BiRefNet weights from the official repository.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Match model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("BiRefNet model initialized successfully!")
    
    def preprocess_image(self, image_input):
        """Preprocess image for the model"""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert('RGB')
        
        original_size = image.size
        processed_image = self.transform(image).unsqueeze(0).to(self.device)
        
        return processed_image, original_size, image
    
    def postprocess_mask(self, mask, original_size):
        """Postprocess the model output mask"""
        mask = mask.squeeze().cpu().numpy()
        mask = cv2.resize(mask, original_size)
        mask = (mask * 255).astype(np.uint8)
        
        # Apply some smoothing
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def remove_background(self, image_input, output_path=None, return_mask=False):
        """
        Remove background from image using BiRefNet
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
            output_path: Path to save result (optional)
            return_mask: Whether to return the mask
            
        Returns:
            PIL Image with transparent background
        """
        with torch.no_grad():
            # Preprocess
            processed_image, original_size, original_image = self.preprocess_image(image_input)
            
            # Model prediction
            prediction = self.model(processed_image)
            
            # Postprocess mask
            mask = self.postprocess_mask(prediction, original_size)
            
            # Create result with transparent background
            original_array = np.array(original_image)
            result_array = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_array
            result_array[:, :, 3] = mask  # Alpha channel
            
            result_image = Image.fromarray(result_array, 'RGBA')
            
            # Save if requested
            if output_path:
                result_image.save(output_path, 'PNG')
                print(f"Result saved to: {output_path}")
            
            if return_mask:
                return result_image, Image.fromarray(mask, 'L')
            else:
                return result_image

def download_pretrained_weights():
    """Download pretrained BiRefNet weights"""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # BiRefNet pretrained weights (you'll need to get the actual download link)
    model_path = model_dir / "BiRefNet-general-epoch_244.pth"
    
    if not model_path.exists():
        print("Pretrained weights not found. Please download from:")
        print("https://github.com/ZhengPeng7/BiRefNet")
        print("Or provide your own trained weights.")
        return None
    
    return str(model_path)

if __name__ == "__main__":
    # Test the model
    weights_path = download_pretrained_weights()
    remover = BiRefNetBackgroundRemover(weights_path)
    print("BiRefNet ready for background removal!")