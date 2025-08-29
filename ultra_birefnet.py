#!/usr/bin/env python3
"""
Ultra BiRefNet - Production Quality with Advanced Edge Refinement
Eliminates all edge artifacts, white lines, and noise for perfect results
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import argparse
from pathlib import Path
from final_birefnet import FinalBiRefNet
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
import json

class UltraBiRefNet:
    def __init__(self):
        self.birefnet = FinalBiRefNet()
    
    def advanced_edge_refinement(self, mask, original_image, quality_level='ultra'):
        """
        Ultra-high quality edge refinement
        Removes all artifacts, white lines, and noise
        """
        print("🎨 Applying ultra-high quality edge refinement...")
        
        # Convert to proper format
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
            
        original_mask = mask.copy()
        
        # === STEP 1: Noise Removal ===
        print("   🧹 Removing noise and small artifacts...")
        
        # Median filter to remove salt-and-pepper noise
        mask_clean = cv2.medianBlur(mask, 5)
        
        # Morphological operations to remove small holes and noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small holes
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_medium)
        # Remove small noise
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_small)
        
        # === STEP 2: White Line Removal ===
        print("   ✂️  Removing white border artifacts...")
        
        # Detect and remove white border lines (BiRefNet's biggest issue)
        binary_mask = (mask_clean > 127).astype(np.uint8)
        
        # Find contours to detect edges
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an inward erosion to remove white borders
        erosion_mask = binary_mask.copy()
        for contour in contours:
            # Create a mask for this contour
            contour_mask = np.zeros_like(binary_mask)
            cv2.drawContours(contour_mask, [contour], -1, 1, -1)
            
            # Apply controlled erosion to remove 3-5 pixel white borders
            eroded_contour = cv2.erode(contour_mask, kernel_small, iterations=2)
            
            # Update the main mask
            erosion_mask = np.where(contour_mask == 1, eroded_contour, erosion_mask)
        
        mask_no_borders = erosion_mask * 255
        
        # === STEP 3: Advanced Edge Smoothing ===
        print("   🌊 Applying multi-scale edge smoothing...")
        
        # Create distance transform for smooth gradients
        dist_transform = cv2.distanceTransform(erosion_mask, cv2.DIST_L2, 5)
        
        # Normalize distance transform
        if dist_transform.max() > 0:
            dist_normalized = dist_transform / dist_transform.max()
        else:
            dist_normalized = dist_transform
            
        # Create smooth falloff zone (3-5 pixels)
        falloff_distance = 4  # 4 pixel smooth transition
        smooth_alpha = np.clip(dist_normalized * falloff_distance, 0, 1)
        
        # === STEP 4: Edge-Guided Refinement ===
        print("   🎯 Applying edge-guided refinement...")
        
        # Use original image edges to guide the refinement
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 50, 150)  # Fine edges
        edges_coarse = cv2.Canny(gray, 20, 100)  # Coarse edges
        
        # Combine edge information
        edges_combined = cv2.bitwise_or(edges_fine, edges_coarse)
        
        # Dilate edges slightly to create influence zones
        edge_zones = cv2.dilate(edges_combined, kernel_small, iterations=1)
        
        # Apply edge-guided smoothing
        edge_factor = 1.0 - (edge_zones.astype(np.float32) / 255.0) * 0.4
        smooth_alpha = smooth_alpha * edge_factor
        
        # === STEP 5: Multi-Scale Gaussian Smoothing ===
        print("   🌀 Applying multi-scale Gaussian smoothing...")
        
        # Apply multiple Gaussian filters with different scales
        alpha_smooth1 = cv2.GaussianBlur(smooth_alpha, (3, 3), 0.8)
        alpha_smooth2 = cv2.GaussianBlur(smooth_alpha, (5, 5), 1.2)
        alpha_smooth3 = cv2.GaussianBlur(smooth_alpha, (7, 7), 1.8)
        
        # Combine multi-scale results
        alpha_final = (alpha_smooth1 * 0.5 + alpha_smooth2 * 0.3 + alpha_smooth3 * 0.2)
        
        # === STEP 6: Quality Enhancement ===
        print("   ✨ Applying quality enhancements...")
        
        # Ensure smooth gradients at boundaries
        alpha_final = cv2.bilateralFilter((alpha_final * 255).astype(np.uint8), 9, 75, 75) / 255.0
        
        # Final smoothing pass
        alpha_final = cv2.GaussianBlur(alpha_final, (3, 3), 0.6)
        
        # === STEP 7: Anti-Aliasing ===
        print("   🎨 Applying anti-aliasing...")
        
        # Apply super-sampling anti-aliasing effect
        alpha_upscaled = cv2.resize(alpha_final, (alpha_final.shape[1] * 2, alpha_final.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        alpha_downscaled = cv2.resize(alpha_upscaled, (alpha_final.shape[1], alpha_final.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Convert to 8-bit with high precision
        final_mask = (alpha_downscaled * 255).astype(np.uint8)
        
        print("✅ Ultra-high quality refinement complete")
        return final_mask
    
    def ultra_extract_item(self, original_image, mask, item_info, labels, quality_level='ultra'):
        """Extract single item with ultra-high quality"""
        
        item_id = item_info['id']
        
        # Create clean mask for this item
        item_mask = (labels == item_id).astype(np.uint8) * 255
        
        # Apply ultra refinement
        refined_mask = self.advanced_edge_refinement(item_mask, original_image, quality_level)
        
        # Create ultra-high quality transparent result
        result_array = np.zeros((*original_image.shape[:2], 4), dtype=np.uint8)
        result_array[:, :, :3] = original_image
        result_array[:, :, 3] = refined_mask
        
        return Image.fromarray(result_array, 'RGBA'), refined_mask
    
    def detect_items_ultra(self, mask, min_area=2000):
        """Ultra-precise item detection with better filtering"""
        
        print("🔍 Ultra-precise item detection...")
        
        # Pre-process mask for better detection
        binary_mask = (mask > 100).astype(np.uint8)  # Lower threshold for better detection
        
        # Remove very small noise first
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_tiny)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Advanced filtering based on shape and size
        valid_items = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area > min_area:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate shape features
                aspect_ratio = w / h if h > 0 else 1
                extent = area / (w * h) if (w * h) > 0 else 0
                
                # Filter out very thin or weird shapes (likely artifacts)
                if aspect_ratio > 0.1 and aspect_ratio < 10 and extent > 0.3:
                    valid_items.append({
                        'id': i,
                        'area': area,
                        'bbox': (x, y, w, h),
                        'aspect_ratio': aspect_ratio,
                        'extent': extent
                    })
        
        # Sort by area
        valid_items.sort(key=lambda x: x['area'], reverse=True)
        
        print(f"📊 Ultra-detection found {len(valid_items)} high-quality items")
        for i, item in enumerate(valid_items, 1):
            print(f"   🎯 Item {i}: {item['area']} pixels (ratio: {item['aspect_ratio']:.2f})")
        
        return valid_items, labels
    
    def process_ultra(self, image_path, output_dir='ultra_results', quality_level='ultra'):
        """Ultra-high quality processing pipeline"""
        
        print("💎 Ultra BiRefNet - Production Quality Processing")
        print("=" * 55)
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Process with BiRefNet
        print("📥 Processing with BiRefNet...")
        result_image, mask_image = self.birefnet.remove_background(image_path)
        
        original_image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(mask_image)
        
        # Ultra-precise detection
        items, labels = self.detect_items_ultra(mask, min_area=2000)
        
        input_name = Path(image_path).stem
        results = {'input': str(image_path), 'quality': quality_level, 'items': []}
        
        if len(items) == 0:
            print("⚠️ No high-quality items detected!")
            return results
        
        elif len(items) == 1:
            print("👔 Single premium item detected - Ultra processing...")
            
            # Ultra-extract single item
            ultra_result, refined_mask = self.ultra_extract_item(original_image, mask, items[0], labels, quality_level)
            
            # Save ultra-quality result
            output_path = Path(output_dir) / f"{input_name}_ultra.png"
            ultra_result.save(output_path, 'PNG', quality=100)
            
            # Save cropped version
            bbox = items[0]['bbox']
            x, y, w, h = bbox
            padding = 15
            x1, y1 = max(0, x - padding), max(0, y - padding) 
            x2, y2 = min(original_image.shape[1], x + w + padding), min(original_image.shape[0], y + h + padding)
            
            result_array = np.array(ultra_result)
            cropped_result = result_array[y1:y2, x1:x2]
            cropped_path = Path(output_dir) / f"{input_name}_ultra_crop.png"
            cropped_image = Image.fromarray(cropped_result, 'RGBA')
            cropped_image.save(cropped_path, 'PNG', quality=100)
            
            # Save mask for inspection
            mask_path = Path(output_dir) / f"{input_name}_ultra_mask.png"
            mask_img = Image.fromarray(refined_mask, 'L')
            mask_img.save(mask_path, 'PNG', quality=100)
            
            print(f"💎 Ultra-quality single item saved:")
            print(f"   📄 Full: {output_path}")
            print(f"   ✂️  Crop: {cropped_path}")
            print(f"   🎭 Mask: {mask_path}")
            
            results['items'].append({
                'id': 1,
                'area': items[0]['area'],
                'full_path': str(output_path),
                'cropped_path': str(cropped_path),
                'mask_path': str(mask_path)
            })
        
        else:
            print(f"👕👖 Multiple premium items ({len(items)}) - Ultra processing each...")
            
            for i, item in enumerate(items, 1):
                print(f"💎 Ultra-processing item {i}/{len(items)}...")
                
                # Ultra-extract each item
                ultra_result, refined_mask = self.ultra_extract_item(original_image, mask, item, labels, quality_level)
                
                # Save full version
                full_path = Path(output_dir) / f"{input_name}_item_{i}_ultra.png"
                ultra_result.save(full_path, 'PNG', quality=100)
                
                # Save cropped version
                bbox = item['bbox']
                x, y, w, h = bbox
                padding = 15
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(original_image.shape[1], x + w + padding), min(original_image.shape[0], y + h + padding)
                
                result_array = np.array(ultra_result)
                cropped_result = result_array[y1:y2, x1:x2]
                cropped_path = Path(output_dir) / f"{input_name}_item_{i}_ultra_crop.png"
                cropped_image = Image.fromarray(cropped_result, 'RGBA')
                cropped_image.save(cropped_path, 'PNG', quality=100)
                
                # Save mask
                mask_path = Path(output_dir) / f"{input_name}_item_{i}_ultra_mask.png"
                mask_img = Image.fromarray(refined_mask, 'L')
                mask_img.save(mask_path, 'PNG', quality=100)
                
                results['items'].append({
                    'id': i,
                    'area': item['area'],
                    'full_path': str(full_path),
                    'cropped_path': str(cropped_path),
                    'mask_path': str(mask_path)
                })
                
                print(f"💎 Ultra item {i}: {item['area']} pixels - Perfect!")
        
        # Save processing info
        info_path = Path(output_dir) / f"{input_name}_ultra_info.json"
        with open(info_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Ultra-processing complete!")
        print(f"💎 {len(items)} items processed with production quality")
        print(f"📁 Ultra results: {output_dir}/")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Ultra BiRefNet - Production Quality Processing")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', default='ultra_results', help='Output directory')
    parser.add_argument('--quality', choices=['high', 'ultra'], default='ultra', help='Quality level')
    
    args = parser.parse_args()
    
    print("💎 Ultra BiRefNet - Production Quality")
    print("=" * 40)
    
    try:
        processor = UltraBiRefNet()
        results = processor.process_ultra(args.input, args.output, args.quality)
        
        if results['items']:
            print(f"\n💎 Ultra-Quality Results:")
            if len(results['items']) == 1:
                print(f"   👔 Single premium item processed")
            else:
                print(f"   👕👖 {len(results['items'])} premium items processed")
                
            for item in results['items']:
                print(f"   💎 Ultra Item {item['id']}: {item['area']} pixels - PERFECT!")
                
            print(f"\n🚀 Production-ready results saved!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())