#!/usr/bin/env python3
"""
Interactive BiRefNet - Select individual clothing items
Click on items to select them individually
"""

import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from pathlib import Path
from final_birefnet import FinalBiRefNet

class InteractiveBiRefNet:
    def __init__(self):
        self.birefnet = FinalBiRefNet()
        self.original_image = None
        self.mask = None
        self.selected_points = []
        self.current_item_mask = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for selecting points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            print(f"🎯 Selected point: ({x}, {y})")
            
            # Draw point on image
            cv2.circle(param['display_image'], (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select Items - Click on clothing items', param['display_image'])
            
    def flood_fill_selection(self, mask, point, tolerance=20):
        """Use flood fill to select connected regions from clicked point"""
        x, y = point
        h, w = mask.shape
        
        # Create binary mask from the segmentation mask
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Check if point is valid
        if x >= w or y >= h or binary_mask[y, x] == 0:
            print(f"⚠️ Point ({x}, {y}) is not on a detected object")
            return None
        
        # Create mask for flood fill
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Perform flood fill from clicked point
        cv2.floodFill(binary_mask, flood_mask, (x, y), 255)
        
        # Extract the filled region
        filled_region = binary_mask.copy()
        
        return filled_region
    
    def connected_component_selection(self, mask, point):
        """Select connected component containing the clicked point"""
        x, y = point
        
        # Convert mask to binary
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        
        # Get label at clicked point
        if x >= labels.shape[1] or y >= labels.shape[0]:
            return None
            
        clicked_label = labels[y, x]
        
        if clicked_label == 0:  # Background
            print(f"⚠️ Clicked on background at ({x}, {y})")
            return None
        
        # Create mask for this component
        component_mask = (labels == clicked_label).astype(np.uint8) * 255
        
        return component_mask
    
    def select_individual_items(self, image_path, output_dir='selected_items'):
        """Interactive selection of individual clothing items"""
        print("🎯 Interactive BiRefNet - Select Individual Items")
        print("=" * 50)
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load and process image with BiRefNet
        print("📥 Processing image with BiRefNet...")
        result_image, mask_image = self.birefnet.remove_background(image_path)
        
        # Load original image
        self.original_image = np.array(Image.open(image_path).convert('RGB'))
        self.mask = np.array(mask_image)
        
        # Create display image
        display_image = self.original_image.copy()
        
        # Setup mouse callback
        cv2.namedWindow('Select Items - Click on clothing items')
        callback_params = {'display_image': display_image}
        cv2.setMouseCallback('Select Items - Click on clothing items', self.mouse_callback, callback_params)
        
        # Show image
        cv2.imshow('Select Items - Click on clothing items', display_image)
        
        print("\n🖱️  Instructions:")
        print("1. Click on clothing items you want to select")
        print("2. Press 'S' to save current selections")
        print("3. Press 'C' to clear selections")
        print("4. Press 'Q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_selected_items(image_path, output_dir)
            elif key == ord('c'):
                self.selected_points = []
                display_image = self.original_image.copy()
                callback_params['display_image'] = display_image
                cv2.imshow('Select Items - Click on clothing items', display_image)
                print("🗑️  Selections cleared")
        
        cv2.destroyAllWindows()
        return self.selected_points
    
    def save_selected_items(self, image_path, output_dir):
        """Save each selected item as separate image"""
        if not self.selected_points:
            print("⚠️ No items selected!")
            return
        
        input_name = Path(image_path).stem
        
        for i, point in enumerate(self.selected_points):
            print(f"💾 Processing selection {i+1}: {point}")
            
            # Get individual item mask using connected components
            item_mask = self.connected_component_selection(self.mask, point)
            
            if item_mask is None:
                print(f"❌ Could not extract item at {point}")
                continue
            
            # Create transparent image for this item
            item_result = np.zeros((*self.original_image.shape[:2], 4), dtype=np.uint8)
            item_result[:, :, :3] = self.original_image
            item_result[:, :, 3] = item_mask
            
            # Save item
            item_image = Image.fromarray(item_result, 'RGBA')
            output_path = Path(output_dir) / f"{input_name}_item_{i+1}.png"
            item_image.save(output_path)
            
            print(f"✅ Saved: {output_path}")
        
        print(f"🎉 Saved {len(self.selected_points)} individual items to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Interactive BiRefNet - Select individual clothing items")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', default='selected_items', help='Output directory')
    
    args = parser.parse_args()
    
    print("🎯 Interactive BiRefNet - Individual Item Selection")
    print("=" * 55)
    
    try:
        selector = InteractiveBiRefNet()
        selected_points = selector.select_individual_items(args.input, args.output)
        
        print(f"\n🎉 Selection complete!")
        print(f"📍 Selected {len(selected_points)} points")
        print(f"📁 Items saved to: {args.output}/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())