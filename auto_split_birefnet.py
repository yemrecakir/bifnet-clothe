#!/usr/bin/env python3
"""
Auto Split BiRefNet - Automatically detect and separate individual clothing items
"""

import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from pathlib import Path
from final_birefnet import FinalBiRefNet
from scipy import ndimage

class AutoSplitBiRefNet:
    def __init__(self):
        self.birefnet = FinalBiRefNet()
        
    def auto_split_items(self, image_path, output_dir='auto_split_items', min_area=5000):
        """Automatically detect and split individual clothing items"""
        print("🔍 Auto Split BiRefNet - Automatically detect individual items")
        print("=" * 60)
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load and process image with BiRefNet
        print("📥 Processing image with BiRefNet...")
        result_image, mask_image = self.birefnet.remove_background(image_path)
        
        # Load original image and mask
        original_image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(mask_image)
        
        # Find connected components
        print("🔍 Detecting individual items...")
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Use connected components to find individual items
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        print(f"📊 Found {num_labels - 1} potential items")
        
        # Filter components by area
        valid_components = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                valid_components.append(i)
                print(f"✅ Item {len(valid_components)}: Area = {area} pixels")
        
        if not valid_components:
            print("⚠️ No valid items found! Try lowering min_area parameter")
            return []
        
        # Save each component as separate item
        input_name = Path(image_path).stem
        saved_items = []
        
        for idx, component_id in enumerate(valid_components, 1):
            print(f"💾 Processing item {idx}/{len(valid_components)}...")
            
            # Create mask for this component
            component_mask = (labels == component_id).astype(np.uint8) * 255
            
            # Apply morphological operations to clean the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel)
            component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_OPEN, kernel)
            
            # Get bounding box for this component
            y_coords, x_coords = np.where(component_mask > 0)
            if len(x_coords) == 0 or len(y_coords) == 0:
                continue
                
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(original_image.shape[1], x_max + padding)
            y_max = min(original_image.shape[0], y_max + padding)
            
            # Create transparent image for this item (full size)
            item_result_full = np.zeros((*original_image.shape[:2], 4), dtype=np.uint8)
            item_result_full[:, :, :3] = original_image
            item_result_full[:, :, 3] = component_mask
            
            # Create cropped version
            item_result_cropped = item_result_full[y_min:y_max, x_min:x_max]
            
            # Save full version
            full_path = Path(output_dir) / f"{input_name}_item_{idx}_full.png"
            item_image_full = Image.fromarray(item_result_full, 'RGBA')
            item_image_full.save(full_path)
            
            # Save cropped version
            cropped_path = Path(output_dir) / f"{input_name}_item_{idx}_cropped.png"
            item_image_cropped = Image.fromarray(item_result_cropped, 'RGBA')
            item_image_cropped.save(cropped_path)
            
            # Save mask for debugging
            mask_path = Path(output_dir) / f"{input_name}_item_{idx}_mask.png"
            mask_image = Image.fromarray(component_mask, 'L')
            mask_image.save(mask_path)
            
            saved_items.append({
                'id': idx,
                'full_path': full_path,
                'cropped_path': cropped_path,
                'mask_path': mask_path,
                'bbox': (x_min, y_min, x_max, y_max),
                'area': stats[component_id, cv2.CC_STAT_AREA]
            })
            
            print(f"✅ Saved item {idx}:")
            print(f"   📄 Full: {full_path}")
            print(f"   ✂️  Cropped: {cropped_path}")
            print(f"   🎭 Mask: {mask_path}")
        
        # Create summary image showing all detected items
        self.create_summary_image(original_image, labels, valid_components, 
                                output_dir, input_name)
        
        print(f"\n🎉 Auto-split complete!")
        print(f"📊 Successfully extracted {len(saved_items)} individual items")
        print(f"📁 Items saved to: {output_dir}/")
        
        return saved_items
    
    def create_summary_image(self, original_image, labels, valid_components, output_dir, input_name):
        """Create a summary image showing all detected items with labels"""
        summary_image = original_image.copy()
        
        # Generate colors for each component
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0),
            (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
        ]
        
        for idx, component_id in enumerate(valid_components):
            color = colors[idx % len(colors)]
            
            # Get component mask
            component_mask = (labels == component_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            cv2.drawContours(summary_image, contours, -1, color, 2)
            
            # Add label
            y_coords, x_coords = np.where(component_mask > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = int(x_coords.mean())
                center_y = int(y_coords.mean())
                
                cv2.putText(summary_image, f'Item {idx + 1}', 
                           (center_x - 30, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Save summary
        summary_path = Path(output_dir) / f"{input_name}_summary.png"
        summary_pil = Image.fromarray(summary_image)
        summary_pil.save(summary_path)
        print(f"📋 Summary image saved: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Auto Split BiRefNet - Automatically detect individual items")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', default='auto_split_items', help='Output directory')
    parser.add_argument('--min-area', type=int, default=5000, help='Minimum area for valid items (default: 5000)')
    
    args = parser.parse_args()
    
    print("🔍 Auto Split BiRefNet - Individual Item Detection")
    print("=" * 55)
    
    try:
        splitter = AutoSplitBiRefNet()
        items = splitter.auto_split_items(args.input, args.output, args.min_area)
        
        if items:
            print(f"\n📊 Detection Results:")
            for item in items:
                print(f"   🎯 Item {item['id']}: {item['area']} pixels")
            
            print(f"\n🎉 Success! Check {args.output}/ for results")
        else:
            print("\n⚠️ No items detected. Try:")
            print("   • Lowering --min-area parameter")
            print("   • Using higher quality input image")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())