#!/usr/bin/env python3
"""
Setup script for BiRefNet Clothing Background Remover
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def download_model_weights():
    """Instructions for downloading model weights"""
    print("\n🤖 Model Weights Setup")
    print("=" * 40)
    print("To get the best results, you need to download BiRefNet pretrained weights.")
    print("\nOptions:")
    print("1. Visit: https://github.com/ZhengPeng7/BiRefNet")
    print("2. Download model weights from the releases section")
    print("3. Place the .pth file in the 'models/' folder")
    print("\nAlternatively, you can use the model without pretrained weights")
    print("(using only backbone pretrained weights)")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"\n📁 Created models directory: {models_dir.absolute()}")

def create_test_script():
    """Create a simple test script"""
    test_script = '''#!/usr/bin/env python3
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
        
        print("\\n📖 Usage examples:")
        print("# Single image:")
        print("python clothing_bg_remover.py your_clothing_image.jpg")
        print("\\n# Folder of images:")
        print("python clothing_bg_remover.py /path/to/clothing/folder/")
        print("\\n# With custom output:")
        print("python clothing_bg_remover.py image.jpg -o result.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        print("\\n💡 Tips:")
        print("- Make sure PyTorch is installed correctly")
        print("- Check if CUDA is available if you want GPU acceleration")
        print("- Download pretrained weights for better results")
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
'''
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test script: test_setup.py")

def main():
    """Main setup function"""
    print("🎽 BiRefNet Clothing Background Remover Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Download model weights info
    download_model_weights()
    
    # Create test script
    create_test_script()
    
    print("\n🎉 Setup complete!")
    print("\n🚀 Next steps:")
    print("1. Run: python test_setup.py")
    print("2. (Optional) Download pretrained weights to models/ folder")
    print("3. Start removing backgrounds: python clothing_bg_remover.py your_image.jpg")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())