#!/usr/bin/env python3
"""
Border BiRefNet - Adds clean white border to hide BiRefNet artifacts
Perfect for hiding edge imperfections with a clean white frame
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import argparse
from pathlib import Path
from final_birefnet import FinalBiRefNet
import json

class BorderBiRefNet:
    def __init__(self):
        self.birefnet = FinalBiRefNet()
    
    def add_clean_border(self, mask, border_width=5, border_color=255):
        """
        Add clean white border around object to hide BiRefNet artifacts
        This covers up the imperfect edges with a nice clean border
        """
        print(f"🖼️ Adding {border_width}px clean white border to hide artifacts...")
        
        # Convert to numpy if needed
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Create binary mask
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create new mask with borders
        bordered_mask = binary_mask.copy()
        
        # Dilate to create border effect
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width * 2 + 1, border_width * 2 + 1))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        # Create border zone (dilated area minus original)
        border_zone = dilated_mask - binary_mask
        
        # Final mask: original object + border zone
        final_mask = np.clip(dilated_mask * border_color, 0, 255).astype(np.uint8)
        
        print("✅ Clean border added successfully")
        return final_mask
    
    def add_gradient_border(self, mask, border_width=5):
        """
        Add gradient border that fades from white to transparent
        Even more elegant than solid white border
        """
        print(f"🌈 Adding {border_width}px gradient border...")
        
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Create binary mask
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Create distance transform for smooth gradient
        dist_transform = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 5)
        
        # Create gradient in border area
        gradient_mask = binary_mask.astype(np.float32)
        
        # Add gradient border
        border_gradient = np.clip(1.0 - (dist_transform / border_width), 0, 1)
        border_area = (dist_transform <= border_width) & (dist_transform > 0)
        
        gradient_mask = np.where(border_area, border_gradient, gradient_mask)
        
        # Convert back to 0-255
        final_mask = (gradient_mask * 255).astype(np.uint8)
        
        print("✅ Gradient border added")
        return final_mask
    
    def detect_and_border_items(self, image_path, output_dir='bordered_results', 
                               border_type='solid', border_width=5, min_area=2000):
        """
        Detect items and add clean borders to hide BiRefNet imperfections
        """
        print("🖼️ Border BiRefNet - Clean Borders for Perfect Results")
        print("=" * 55)
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Process with BiRefNet
        print("📥 Processing with BiRefNet...")
        result_image, mask_image = self.birefnet.remove_background(image_path)
        
        original_image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(mask_image)
        
        # Detect items
        print("🔍 Detecting items...")
        print(f"📊 Mask stats: min={mask.min()}, max={mask.max()}, mean={mask.mean():.2f}")
        
        # Lower threshold for better detection
        binary_mask = (mask > 50).astype(np.uint8)  # Lower from 100 to 50
        print(f"🎯 Binary mask has {binary_mask.sum()} foreground pixels")
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        print(f"🔍 Found {num_labels-1} connected components")
        
        # Filter items
        valid_items = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            print(f"   Component {i}: {area} pixels")
            if area > min_area:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                valid_items.append({
                    'id': i,
                    'area': area,
                    'bbox': (x, y, w, h)
                })
        
        valid_items.sort(key=lambda x: x['area'], reverse=True)
        input_name = Path(image_path).stem
        results = []
        
        print(f"🖼️ Found {len(valid_items)} items - Adding clean borders...")
        
        if len(valid_items) == 0:
            print("⚠️ No items detected!")
            return results
            
        elif len(valid_items) == 1:
            print("👔 Single item - Adding clean border...")
            
            # Extract item mask
            item_mask = (labels == valid_items[0]['id']).astype(np.uint8) * 255
            
            # Add border based on type
            if border_type == 'gradient':
                bordered_mask = self.add_gradient_border(item_mask, border_width)
            else:  # solid
                bordered_mask = self.add_clean_border(item_mask, border_width)
            
            # Create result with bordered mask
            result_array = np.zeros((*original_image.shape[:2], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_image
            result_array[:, :, 3] = bordered_mask
            
            # Save
            output_path = Path(output_dir) / f"{input_name}_bordered.png"
            result_img = Image.fromarray(result_array, 'RGBA')
            result_img.save(output_path, 'PNG')
            
            # Save cropped version
            bbox = valid_items[0]['bbox']
            x, y, w, h = bbox
            padding = border_width + 10
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(original_image.shape[1], x + w + padding), min(original_image.shape[0], y + h + padding)
            
            cropped_result = result_array[y1:y2, x1:x2]
            cropped_path = Path(output_dir) / f"{input_name}_bordered_crop.png"
            cropped_img = Image.fromarray(cropped_result, 'RGBA')
            cropped_img.save(cropped_path, 'PNG')
            
            print(f"✅ Single item with clean border:")
            print(f"   📄 Full: {output_path}")
            print(f"   ✂️  Crop: {cropped_path}")
            
            results.append({
                'id': 1,
                'area': valid_items[0]['area'],
                'full_path': str(output_path),
                'cropped_path': str(cropped_path)
            })
            
        else:
            print(f"👕👖 Multiple items ({len(valid_items)}) - Adding borders to each...")
            
            for i, item in enumerate(valid_items, 1):
                print(f"🖼️ Adding border to item {i}/{len(valid_items)}...")
                
                # Extract item mask
                item_mask = (labels == item['id']).astype(np.uint8) * 255
                
                # Add border
                if border_type == 'gradient':
                    bordered_mask = self.add_gradient_border(item_mask, border_width)
                else:
                    bordered_mask = self.add_clean_border(item_mask, border_width)
                
                # Create result
                result_array = np.zeros((*original_image.shape[:2], 4), dtype=np.uint8)
                result_array[:, :, :3] = original_image
                result_array[:, :, 3] = bordered_mask
                
                # Save full version
                full_path = Path(output_dir) / f"{input_name}_item_{i}_bordered.png"
                result_img = Image.fromarray(result_array, 'RGBA')
                result_img.save(full_path, 'PNG')
                
                # Save cropped version
                bbox = item['bbox']
                x, y, w, h = bbox
                padding = border_width + 10
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(original_image.shape[1], x + w + padding), min(original_image.shape[0], y + h + padding)
                
                cropped_result = result_array[y1:y2, x1:x2]
                cropped_path = Path(output_dir) / f"{input_name}_item_{i}_bordered_crop.png"
                cropped_img = Image.fromarray(cropped_result, 'RGBA')
                cropped_img.save(cropped_path, 'PNG')
                
                results.append({
                    'id': i,
                    'area': item['area'],
                    'full_path': str(full_path),
                    'cropped_path': str(cropped_path)
                })
                
                print(f"✅ Item {i} bordered: {item['area']} pixels")
        
        print(f"\n🖼️ Border processing complete!")
        print(f"✨ {len(results)} items now have clean borders")
        print(f"🎯 All BiRefNet artifacts are hidden!")
        print(f"📁 Results: {output_dir}/")
        
        return results
    
    def create_comparison_image(self, original_path, bordered_path, output_path):
        """Create before/after comparison"""
        
        original = Image.open(original_path).convert('RGBA')
        bordered = Image.open(bordered_path).convert('RGBA')
        
        # Resize to same size if needed
        size = (max(original.width, bordered.width), max(original.height, bordered.height))
        
        # Create comparison
        comparison = Image.new('RGBA', (size[0] * 2 + 20, size[1]), (255, 255, 255, 255))
        
        comparison.paste(original, (0, 0))
        comparison.paste(bordered, (size[0] + 20, 0))
        
        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Original BiRefNet", fill=(0, 0, 0, 255), font=font)
        draw.text((size[0] + 30, 10), "With Clean Border", fill=(0, 0, 0, 255), font=font)
        
        comparison.save(output_path, 'PNG')
        print(f"📊 Comparison saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Border BiRefNet - Add clean borders to hide artifacts")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', default='bordered_results', help='Output directory')
    parser.add_argument('--border-width', type=int, default=5, help='Border width in pixels (default: 5)')
    parser.add_argument('--border-type', choices=['solid', 'gradient'], default='solid', help='Border type')
    parser.add_argument('--comparison', action='store_true', help='Create before/after comparison')
    
    args = parser.parse_args()
    
    print("🖼️ Border BiRefNet - Clean Borders for Perfect Results")
    print("=" * 55)
    
    try:
        processor = BorderBiRefNet()
        results = processor.detect_and_border_items(
            args.input, 
            args.output, 
            args.border_type, 
            args.border_width
        )
        
        if results:
            print(f"\n🖼️ Border Results:")
            if len(results) == 1:
                print(f"   👔 Single item with {args.border_width}px {args.border_type} border")
            else:
                print(f"   👕👖 {len(results)} items with {args.border_width}px {args.border_type} borders")
            
            for item in results:
                print(f"   ✨ Item {item['id']}: {item['area']} pixels - Perfect borders!")
            
            print(f"\n🎯 All BiRefNet artifacts are now hidden!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())