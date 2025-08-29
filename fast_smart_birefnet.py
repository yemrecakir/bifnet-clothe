#!/usr/bin/env python3
"""
Fast Smart BiRefNet - Optimized version with quick edge refinement
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import argparse
from pathlib import Path
from final_birefnet import FinalBiRefNet
import json

class FastSmartBiRefNet:
    def __init__(self):
        self.birefnet = FinalBiRefNet()
    
    def quick_edge_refinement(self, mask, blur_radius=1):
        """Fast edge refinement to remove BiRefNet white borders"""
        
        # Convert to numpy if needed
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Quick cleanup
        mask_cleaned = cv2.medianBlur(mask, 3)
        
        # Remove white border artifacts by slight erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_eroded = cv2.erode(mask_cleaned, kernel, iterations=1)
        
        # Smooth edges with small blur
        mask_float = mask_eroded.astype(np.float32) / 255.0
        mask_smooth = cv2.GaussianBlur(mask_float, (blur_radius * 2 + 1, blur_radius * 2 + 1), blur_radius)
        
        # Convert back
        refined_mask = (mask_smooth * 255).astype(np.uint8)
        
        return refined_mask
    
    def fast_detect_items(self, mask, min_area=3000):
        """Fast item detection"""
        
        binary_mask = (mask > 127).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        
        valid_items = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                valid_items.append({
                    'id': i,
                    'area': area,
                    'bbox': (x, y, w, h)
                })
        
        # Sort by area (largest first)
        valid_items.sort(key=lambda x: x['area'], reverse=True)
        return valid_items, labels
    
    def process_smart(self, image_path, output_dir='fast_results'):
        """Fast smart processing"""
        
        print("⚡ Fast Smart BiRefNet")
        print("=" * 30)
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Process with BiRefNet
        print("📥 Processing...")
        result_image, mask_image = self.birefnet.remove_background(image_path)
        
        original_image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(mask_image)
        
        # Detect items
        items, labels = self.fast_detect_items(mask)
        input_name = Path(image_path).stem
        
        print(f"🔍 Detected {len(items)} items")
        
        if len(items) == 0:
            print("⚠️ No items detected!")
            return []
        
        elif len(items) == 1:
            print("👔 Single item - Direct processing...")
            
            # Refine edges for single item
            item_mask = (labels == items[0]['id']).astype(np.uint8) * 255
            refined_mask = self.quick_edge_refinement(item_mask)
            
            # Create result
            result_array = np.zeros((*original_image.shape[:2], 4), dtype=np.uint8)
            result_array[:, :, :3] = original_image
            result_array[:, :, 3] = refined_mask
            
            # Save
            output_path = Path(output_dir) / f"{input_name}_refined.png"
            result_img = Image.fromarray(result_array, 'RGBA')
            result_img.save(output_path)
            
            print(f"✅ Saved: {output_path}")
            return [{'path': output_path, 'area': items[0]['area']}]
        
        else:
            print(f"👕👖 Multiple items ({len(items)}) - Processing each...")
            
            results = []
            for i, item in enumerate(items, 1):
                # Extract and refine each item
                item_mask = (labels == item['id']).astype(np.uint8) * 255
                refined_mask = self.quick_edge_refinement(item_mask)
                
                # Create result
                result_array = np.zeros((*original_image.shape[:2], 4), dtype=np.uint8)
                result_array[:, :, :3] = original_image
                result_array[:, :, 3] = refined_mask
                
                # Save full version
                full_path = Path(output_dir) / f"{input_name}_item_{i}.png"
                result_img = Image.fromarray(result_array, 'RGBA')
                result_img.save(full_path)
                
                # Save cropped version
                x, y, w, h = item['bbox']
                padding = 10
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(original_image.shape[1], x + w + padding), min(original_image.shape[0], y + h + padding)
                
                cropped_result = result_array[y1:y2, x1:x2]
                cropped_path = Path(output_dir) / f"{input_name}_item_{i}_crop.png"
                cropped_img = Image.fromarray(cropped_result, 'RGBA')
                cropped_img.save(cropped_path)
                
                results.append({
                    'id': i,
                    'full_path': full_path,
                    'cropped_path': cropped_path,
                    'area': item['area']
                })
                
                print(f"✅ Item {i}: {item['area']} pixels")
            
            print(f"🎉 Saved {len(results)} items to {output_dir}/")
            return results

def main():
    parser = argparse.ArgumentParser(description="Fast Smart BiRefNet")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', default='fast_results', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        processor = FastSmartBiRefNet()
        results = processor.process_smart(args.input, args.output)
        
        if results:
            if len(results) == 1:
                print(f"✅ Single item processed")
            else:
                print(f"✅ {len(results)} items processed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())