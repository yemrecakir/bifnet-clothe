import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from transformers import AutoModelForImageSegmentation
import requests
from io import BytesIO

class BiRefNetBackgroundRemover:
    def __init__(self, model_name="ZhengPeng7/BiRefNet_T_2K"):
        """
        Initialize BiRefNet model for background removal
        Using the high-resolution variant for better quality
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            # Load the BiRefNet model
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Model {model_name} loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative model...")
            try:
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet",
                    trust_remote_code=True
                )
                self.model.to(self.device)
                self.model.eval()
                print("Alternative BiRefNet model loaded successfully!")
            except Exception as e2:
                print(f"Error loading alternative model: {e2}")
                raise e2
        
        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # HR variant uses higher resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess image for the model
        """
        if isinstance(image, str):
            # If image is a file path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # If image is numpy array (from OpenCV)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        original_size = image.size
        processed_image = self.transform(image).unsqueeze(0).to(self.device)
        
        return processed_image, original_size
    
    def postprocess_mask(self, mask, original_size):
        """
        Postprocess the model output mask
        """
        # Convert to numpy and resize to original size
        mask = mask.squeeze().cpu().numpy()
        mask = cv2.resize(mask, original_size)
        
        # Threshold to create binary mask
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        return mask
    
    def remove_background(self, image_input, output_path=None, return_mask=False):
        """
        Remove background from image using BiRefNet
        
        Args:
            image_input: Can be file path, PIL Image, or numpy array
            output_path: Path to save the result (optional)
            return_mask: Whether to return the mask along with the result
            
        Returns:
            PIL Image with transparent background (and optionally the mask)
        """
        with torch.no_grad():
            # Preprocess
            processed_image, original_size = self.preprocess_image(image_input)
            
            # Get prediction
            prediction = self.model(processed_image)
            
            # Handle different output formats
            if isinstance(prediction, tuple):
                mask = prediction[0]
            elif isinstance(prediction, dict):
                mask = prediction.get('prediction', prediction.get('logits', prediction))
            else:
                mask = prediction
            
            # Apply sigmoid and postprocess
            mask = torch.sigmoid(mask)
            mask = self.postprocess_mask(mask, original_size)
            
            # Load original image
            if isinstance(image_input, str):
                original_image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                original_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                original_image = image_input.convert('RGB')
            
            # Apply mask to create transparent background
            original_array = np.array(original_image)
            
            # Create RGBA image
            result_array = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_array
            result_array[:, :, 3] = mask  # Alpha channel
            
            result_image = Image.fromarray(result_array, 'RGBA')
            
            # Save if output path is provided
            if output_path:
                result_image.save(output_path, 'PNG')
                print(f"Result saved to: {output_path}")
            
            if return_mask:
                return result_image, Image.fromarray(mask, 'L')
            else:
                return result_image
    
    def batch_remove_background(self, image_paths, output_dir):
        """
        Remove background from multiple images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{filename}_no_bg.png")
            
            try:
                result = self.remove_background(image_path, output_path)
                results.append(result)
                print(f"✓ Processed: {filename}")
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
                results.append(None)
        
        return results

def main():
    """
    Test function with example usage
    """
    print("BiRefNet Background Remover - Clothing Edition")
    print("=" * 50)
    
    # Initialize the model
    try:
        remover = BiRefNetBackgroundRemover()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    # Example usage
    print("\nModel ready! You can now use it to remove backgrounds from clothing images.")
    print("\nExample usage:")
    print("remover.remove_background('path/to/your/clothing/image.jpg', 'output_no_bg.png')")
    
    return remover

if __name__ == "__main__":
    remover = main()