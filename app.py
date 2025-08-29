#!/usr/bin/env python3
"""
Web API for BiRefNet background removal
"""

from flask import Flask, request, jsonify, send_file
import os
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

class BiRefNetService:
    def __init__(self):
        # Import heavy dependencies only when needed
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForImageSegmentation
        import cv2
        
        self.torch = torch
        self.F = F
        self.cv2 = cv2
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        
        print("📥 Loading BiRefNet...")
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                'ZhengPeng7/BiRefNet',
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            print("✅ BiRefNet loaded!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e
        
        self.image_size = 1024
    
    def preprocess_image(self, image):
        original_size = image.size
        image_resized = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        image_tensor = self.torch.tensor(np.array(image_resized)).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0
        
        mean = self.torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = self.torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor, original_size, image
    
    def remove_background(self, image):
        with self.torch.no_grad():
            processed_image, original_size, original_image = self.preprocess_image(image)
            
            prediction = self.model(processed_image)
            
            if isinstance(prediction, dict):
                pred_mask = prediction.get('logits', prediction.get('prediction', list(prediction.values())[0]))
            elif isinstance(prediction, (list, tuple)):
                pred_mask = prediction[0]
            else:
                pred_mask = prediction
            
            pred_mask = self.torch.sigmoid(pred_mask)
            pred_mask = self.F.interpolate(
                pred_mask,
                size=original_size[::-1],
                mode='bilinear',
                align_corners=False
            )
            
            mask_np = pred_mask.squeeze().cpu().numpy()
            mask_np = self.post_process_mask(mask_np)
            
            original_array = np.array(original_image)
            result_array = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_array
            result_array[:, :, 3] = (mask_np * 255).astype(np.uint8)
            
            result_image = Image.fromarray(result_array, 'RGBA')
            return result_image
    
    def post_process_mask(self, mask):
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = self.cv2.morphologyEx(mask_uint8, self.cv2.MORPH_CLOSE, kernel)
        kernel_small = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = self.cv2.morphologyEx(mask_uint8, self.cv2.MORPH_OPEN, kernel_small)
        mask_uint8 = self.cv2.GaussianBlur(mask_uint8, (3, 3), 0)
        return mask_uint8.astype(np.float32) / 255.0

# Global service instance
birefnet_service = None

@app.route('/')
def health_check():
    return {'status': 'ok', 'service': 'BiRefNet Background Remover'}

@app.route('/init', methods=['POST'])
def init_model():
    try:
        global birefnet_service
        if birefnet_service is None:
            print("🔄 Initializing BiRefNet model...")
            birefnet_service = BiRefNetService()
            return {'status': 'initialized', 'message': 'Model loaded successfully'}
        else:
            return {'status': 'already_initialized', 'message': 'Model already loaded'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/remove-background', methods=['POST'])
def remove_background():
    try:
        global birefnet_service
        if birefnet_service is None:
            print("🔄 Loading BiRefNet model...")
            birefnet_service = BiRefNetService()
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Remove background
        result = birefnet_service.remove_background(image)
        
        # Return image
        img_io = io.BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)