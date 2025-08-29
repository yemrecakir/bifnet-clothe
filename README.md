# 🔥 BiRefNet Clothes Background Remover

High-quality background removal for clothing and fashion items using BiRefNet (Bilateral Reference for High-Resolution Dichotomous Image Segmentation).

![Demo](https://img.shields.io/badge/Demo-Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.7+-red)

## 🚀 Features

- 🎯 **Specialized for Clothing**: Optimized for fashion items and apparel
- 🖼️ **High Resolution**: Supports up to 1024x1024 processing
- ⚡ **Fast Processing**: ~40 seconds on CPU, ~5-10 seconds on GPU
- 🎭 **Perfect Segmentation**: Clean masks and transparent backgrounds
- 📱 **Easy to Use**: Simple command-line interface
- 🔥 **State-of-the-Art**: Uses real BiRefNet pretrained weights

## 🛠️ Installation

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/markossda/bifnet-clothes.git
cd bifnet-clothes

# Install dependencies and setup
python setup.py

# Test installation
python test_setup.py
```

### Manual Installation
```bash
# Install required packages
pip install -r requirements.txt

# Install additional dependencies
pip install transformers einops kornia
```

## 🎯 Usage

### Basic Usage
```bash
# Process single image
python final_birefnet.py your_clothing_image.jpg

# With custom output
python final_birefnet.py image.jpg -o result.png

# Save mask separately
python final_birefnet.py image.jpg -o result.png -m mask.png
```

### Python API
```python
from final_birefnet import FinalBiRefNet

# Initialize
birefnet = FinalBiRefNet()

# Remove background
result, mask = birefnet.remove_background('clothing_image.jpg', 'output.png')
```

## 📊 Performance

| Device | Processing Time | Quality |
|--------|----------------|---------|
| CPU (Multi-core) | ~40 seconds | Excellent |
| GPU (CUDA) | ~5-10 seconds | Excellent |
| Memory Usage | ~2-4GB | Efficient |

## 🎨 Examples

### Input vs Output
Original clothing images with backgrounds → Clean transparent PNGs

**Supported Items:**
- ✅ Shirts, T-shirts, Tops
- ✅ Pants, Jeans, Bottoms  
- ✅ Dresses, Skirts
- ✅ Jackets, Coats
- ✅ Shoes, Sneakers
- ✅ Accessories (bags, jewelry, etc.)

## 🔧 Advanced Options

### Custom Model Loading
```bash
# Use specific model variant
python final_birefnet.py image.jpg --model ZhengPeng7/BiRefNet_HR
```

### Batch Processing
```python
# Process multiple images
for image_file in image_files:
    result, mask = birefnet.remove_background(image_file)
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.13+
- Transformers
- Pillow
- OpenCV
- NumPy
- Hugging Face Hub

See `requirements.txt` for full dependencies.

## 🏗️ Architecture

BiRefNet uses:
- **Bilateral Reference**: Advanced segmentation technique
- **High-Resolution Processing**: Maintains detail quality
- **Pretrained Weights**: From Hugging Face model hub
- **Multi-Scale Features**: Better edge detection

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Force CPU usage
CUDA_VISIBLE_DEVICES="" python final_birefnet.py image.jpg
```

**"Model loading failed"**
```bash
# Re-download model
python download_birefnet.py
```

**"Poor quality results"**
- Ensure good lighting in input image
- Use high-resolution source images
- Check that clothing is clearly visible

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - Original BiRefNet implementation
- Hugging Face - Model hosting and transformers library
- PyTorch Team - Deep learning framework

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Search existing issues
3. Open a new issue with details

---

**Made with 🔥 for fashion and e-commerce applications**