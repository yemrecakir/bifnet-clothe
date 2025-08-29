#!/usr/bin/env python3
"""
Smart BiRefNet - Intelligent clothing detection with edge refinement
Automatically detects single vs multiple items and refines edges
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import argparse
from pathlib import Path
from final_birefnet import FinalBiRefNet
from scipy import ndimage
import json

class SmartBiRefNet:
    def __init__(self):
        self.birefnet = FinalBiRefNet()
        
    def refine_edges(self, mask, original_image, blur_radius=2, feather_amount=3):
        """Advanced edge refinement to remove BiRefNet artifacts"""
        
        # Convert to numpy if PIL
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        print("🎨 Refining edges to remove artifacts...")
        
        # 1. Remove noise and small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        # 2. Edge detection on original image for guidance
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 3. Distance transform for smooth falloff
        dist_transform = cv2.distanceTransform(mask_cleaned, cv2.DIST_L2, 5)
        
        # 4. Create smooth edge gradient
        dist_normalized = np.clip(dist_transform / feather_amount, 0, 1)
        
        # 5. Apply Gaussian blur for natural edge softening
        mask_float = mask_cleaned.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (blur_radius * 2 + 1, blur_radius * 2 + 1), blur_radius / 2)
        
        # 6. Combine distance-based and blur-based smoothing
        alpha_channel = np.maximum(mask_blurred, dist_normalized)
        
        # 7. Apply edge-guided refinement
        # Reduce opacity near detected edges to create more natural boundaries
        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_factor = 1.0 - (edge_dilated.astype(np.float32) / 255.0) * 0.3
        alpha_channel = alpha_channel * edge_factor
        
        # 8. Final smoothing
        alpha_channel = cv2.GaussianBlur(alpha_channel, (3, 3), 1)
        
        # 9. Remove BiRefNet's typical white border artifacts
        # Detect and fix bright edge artifacts
        mask_binary = (alpha_channel > 0.1).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Create a small inward erosion to remove white borders
            mask_temp = np.zeros_like(mask_binary)
            cv2.drawContours(mask_temp, [contour], -1, 1, -1)
            
            # Erode slightly to remove artifact borders
            eroded = cv2.erode(mask_temp, np.ones((2, 2), np.uint8), iterations=1)
            
            # Apply erosion to alpha channel
            alpha_channel = np.where(mask_temp == 1, 
                                   np.where(eroded == 1, alpha_channel, alpha_channel * 0.7),
                                   alpha_channel)
        
        # Convert back to 0-255 range
        refined_mask = (alpha_channel * 255).astype(np.uint8)
        
        print("✅ Edge refinement complete")
        return refined_mask
    
    def detect_clothing_count(self, mask, min_area=3000):
        """Intelligently detect number of clothing items"""
        
        print("🔍 Analyzing clothing items...")
        
        # Convert to binary
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Filter by area and analyze
        valid_items = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                # Get bounding box info
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate aspect ratio and other features
                aspect_ratio = w / h if h > 0 else 1
                
                valid_items.append({
                    'id': i,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'centroid': centroids[i],
                    'aspect_ratio': aspect_ratio
                })
        
        # Sort by area (largest first)
        valid_items.sort(key=lambda x: x['area'], reverse=True)
        
        print(f"📊 Detected {len(valid_items)} clothing items")
        for i, item in enumerate(valid_items, 1):
            print(f"   🎯 Item {i}: {item['area']} pixels (ratio: {item['aspect_ratio']:.2f})")
        
        return valid_items, labels
    
    def extract_single_item(self, original_image, mask, item_info, labels):
        """Extract a single clothing item with refined edges"""
        
        item_id = item_info['id']
        
        # Create clean mask for this item
        item_mask = (labels == item_id).astype(np.uint8) * 255
        
        # Refine edges specifically for this item
        refined_mask = self.refine_edges(item_mask, original_image)
        
        # Create transparent result
        result_array = np.zeros((*original_image.shape[:2], 4), dtype=np.uint8)
        result_array[:, :, :3] = original_image
        result_array[:, :, 3] = refined_mask
        
        return Image.fromarray(result_array, 'RGBA')
    
    def smart_process(self, image_path, output_dir='smart_results', auto_threshold=0.8):
        """Smart processing: auto-detect and handle single vs multiple items"""
        
        print("🧠 Smart BiRefNet - Intelligent Processing")
        print("=" * 50)
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load and process with BiRefNet
        print("📥 Processing with BiRefNet...")
        result_image, mask_image = self.birefnet.remove_background(image_path)
        
        # Load original image
        original_image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(mask_image)
        
        # Analyze items
        items, labels = self.detect_clothing_count(mask)
        
        input_name = Path(image_path).stem
        results = {'input': str(image_path), 'items': []}
        
        if len(items) == 0:
            print("⚠️ No clothing items detected!")
            return results
        
        elif len(items) == 1:
            print("👔 Single item detected - Processing directly...")
            
            # Extract single item with edge refinement
            refined_result = self.extract_single_item(original_image, mask, items[0], labels)
            
            # Save result
            output_path = Path(output_dir) / f"{input_name}_refined.png"
            refined_result.save(output_path)
            
            # Also save cropped version
            bbox = items[0]['bbox']
            x, y, w, h = bbox
            padding = 20
            x_crop = max(0, x - padding)
            y_crop = max(0, y - padding)
            w_crop = min(original_image.shape[1] - x_crop, w + 2 * padding)
            h_crop = min(original_image.shape[0] - y_crop, h + 2 * padding)
            
            result_array = np.array(refined_result)
            cropped_result = result_array[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
            cropped_image = Image.fromarray(cropped_result, 'RGBA')
            
            cropped_path = Path(output_dir) / f"{input_name}_cropped.png"
            cropped_image.save(cropped_path)
            
            print(f"✅ Single item saved:")
            print(f"   📄 Full: {output_path}")
            print(f"   ✂️  Cropped: {cropped_path}")
            
            results['items'].append({
                'id': 1,
                'area': items[0]['area'],
                'full_path': str(output_path),
                'cropped_path': str(cropped_path),
                'bbox': bbox
            })
            
        else:
            print(f"👕👖 Multiple items detected ({len(items)}) - Processing individually...")
            
            # Create summary image
            summary_image = self.create_detection_summary(original_image, labels, items, output_dir, input_name)
            
            # Process each item
            for i, item in enumerate(items, 1):
                print(f"💾 Processing item {i}/{len(items)}...")
                
                # Extract with edge refinement
                refined_result = self.extract_single_item(original_image, mask, item, labels)
                
                # Save full version
                full_path = Path(output_dir) / f"{input_name}_item_{i}_refined.png"
                refined_result.save(full_path)
                
                # Save cropped version
                bbox = item['bbox']
                x, y, w, h = bbox
                padding = 20
                x_crop = max(0, x - padding)
                y_crop = max(0, y - padding)
                w_crop = min(original_image.shape[1] - x_crop, w + 2 * padding)
                h_crop = min(original_image.shape[0] - y_crop, h + 2 * padding)
                
                result_array = np.array(refined_result)
                cropped_result = result_array[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
                cropped_image = Image.fromarray(cropped_result, 'RGBA')
                
                cropped_path = Path(output_dir) / f"{input_name}_item_{i}_cropped.png"
                cropped_image.save(cropped_path)
                
                results['items'].append({
                    'id': i,
                    'area': item['area'],
                    'full_path': str(full_path),
                    'cropped_path': str(cropped_path),
                    'bbox': bbox
                })
                
                print(f"✅ Item {i} saved with refined edges")
        
        # Save processing info
        info_path = Path(output_dir) / f"{input_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Smart processing complete!")
        print(f"📊 Processed {len(items)} items with edge refinement")
        print(f"📁 Results saved to: {output_dir}/")
        
        return results
    
    def create_detection_summary(self, original_image, labels, items, output_dir, input_name):
        """Create summary image showing detected items"""
        
        summary_image = original_image.copy()
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0)
        ]
        
        for idx, item in enumerate(items):
            color = colors[idx % len(colors)]
            item_id = item['id']
            
            # Draw bounding box
            x, y, w, h = item['bbox']
            cv2.rectangle(summary_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f'Item {idx + 1}'
            cv2.putText(summary_image, label, 
                       (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw contour
            item_mask = (labels == item_id).astype(np.uint8)
            contours, _ = cv2.findContours(item_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(summary_image, contours, -1, color, 1)
        
        # Save summary
        summary_path = Path(output_dir) / f"{input_name}_detection_summary.png"
        summary_pil = Image.fromarray(summary_image)
        summary_pil.save(summary_path)
        
        print(f"📋 Detection summary: {summary_path}")
        return summary_image

def main():
    parser = argparse.ArgumentParser(description="Smart BiRefNet with edge refinement and intelligent detection")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', default='smart_results', help='Output directory')
    parser.add_argument('--min-area', type=int, default=3000, help='Minimum area for valid items')
    
    args = parser.parse_args()
    
    print("🧠 Smart BiRefNet - Intelligent Clothing Processing")
    print("=" * 55)
    
    try:
        processor = SmartBiRefNet()
        results = processor.smart_process(args.input, args.output)
        
        if results['items']:
            print(f"\n📊 Processing Summary:")
            if len(results['items']) == 1:
                print(f"   👔 Single item processed with refined edges")
            else:
                print(f"   👕👖 {len(results['items'])} items processed individually")
            
            for item in results['items']:
                print(f"   🎯 Item {item['id']}: {item['area']} pixels")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())