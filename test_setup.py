#!/usr/bin/env python3
"""
Test script for BiRefNet Clothing Background Remover
"""

import sys
from pathlib import Path
from clothing_bg_remover import ClothingBackgroundRemover

def test_model():
    """Test if the model can be initialized"""
    print("🧪 Testing model initialization...")
    
    try:
        remover = ClothingBackgroundRemover()
        print("✅ Model initialized successfully!")
        print("🎽 Ready to remove backgrounds from clothing images!")
        
        print("\n📖 Usage examples:")
        print("# Single image:")
        print("python clothing_bg_remover.py your_clothing_image.jpg")
        print("\n# Folder of images:")
        print("python clothing_bg_remover.py /path/to/clothing/folder/")
        print("\n# With custom output:")
        print("python clothing_bg_remover.py image.jpg -o result.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        print("\n💡 Tips:")
        print("- Make sure PyTorch is installed correctly")
        print("- Check if CUDA is available if you want GPU acceleration")
        print("- Download pretrained weights for better results")
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
