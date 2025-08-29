#!/usr/bin/env python3
"""
Bounding Box BiRefNet - Draw boxes to select individual items
"""

import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from pathlib import Path
from final_birefnet import FinalBiRefNet

class BBoxBiRefNet:
    def __init__(self):
        self.birefnet = FinalBiRefNet()
        self.original_image = None
        self.mask = None
        self.boxes = []
        self.current_box = None
        self.drawing = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = {'start': (x, y), 'end': (x, y)}
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_box['end'] = (x, y)
            # Redraw image with current box
            self.redraw_image(param['display_image'])
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box['end'] = (x, y)
            self.boxes.append(self.current_box.copy())
            print(f"📦 Box {len(self.boxes)}: {self.current_box['start']} to {self.current_box['end']}")
            self.current_box = None
            self.redraw_image(param['display_image'])
    
    def redraw_image(self, display_image):
        """Redraw image with all boxes"""
        img = self.original_image.copy()
        
        # Draw saved boxes
        for i, box in enumerate(self.boxes):
            cv2.rectangle(img, box['start'], box['end'], (0, 255, 0), 2)
            cv2.putText(img, f'Item {i+1}', 
                       (box['start'][0], box['start'][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current box being drawn
        if self.current_box and self.drawing:
            cv2.rectangle(img, self.current_box['start'], self.current_box['end'], (255, 0, 0), 2)
        
        cv2.imshow('Draw Boxes - Drag to select items', img)
    
    def extract_item_from_box(self, box):
        """Extract item from bounding box region"""
        x1, y1 = box['start']
        x2, y2 = box['end']
        
        # Ensure coordinates are valid
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Crop the region
        mask_region = self.mask[y1:y2, x1:x2]
        image_region = self.original_image[y1:y2, x1:x2]
        
        # Create full-size mask
        full_mask = np.zeros_like(self.mask)
        full_mask[y1:y2, x1:x2] = mask_region
        
        return full_mask, (x1, y1, x2, y2)
    
    def select_with_boxes(self, image_path, output_dir='bbox_items'):
        """Select items using bounding boxes"""
        print("📦 Bounding Box BiRefNet - Draw boxes around items")
        print("=" * 50)
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load and process image with BiRefNet
        print("📥 Processing image with BiRefNet...")
        result_image, mask_image = self.birefnet.remove_background(image_path)
        
        # Load original image
        self.original_image = np.array(Image.open(image_path).convert('RGB'))
        self.mask = np.array(mask_image)
        
        # Setup OpenCV window
        cv2.namedWindow('Draw Boxes - Drag to select items')
        callback_params = {'display_image': self.original_image}
        cv2.setMouseCallback('Draw Boxes - Drag to select items', self.mouse_callback, callback_params)
        
        # Initial display
        self.redraw_image(self.original_image)
        
        print("\n🖱️  Instructions:")
        print("1. Drag mouse to draw bounding boxes around items")
        print("2. Press 'S' to save selected items")
        print("3. Press 'C' to clear all boxes")
        print("4. Press 'U' to undo last box")
        print("5. Press 'Q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_boxed_items(image_path, output_dir)
            elif key == ord('c'):
                self.boxes = []
                self.redraw_image(self.original_image)
                print("🗑️  All boxes cleared")
            elif key == ord('u') and self.boxes:
                removed = self.boxes.pop()
                self.redraw_image(self.original_image)
                print(f"↶ Undid box: {removed}")
        
        cv2.destroyAllWindows()
        return self.boxes
    
    def save_boxed_items(self, image_path, output_dir):
        """Save items from bounding boxes"""
        if not self.boxes:
            print("⚠️ No boxes drawn!")
            return
        
        input_name = Path(image_path).stem
        
        for i, box in enumerate(self.boxes):
            print(f"💾 Processing box {i+1}: {box}")
            
            # Extract item from box
            item_mask, coords = self.extract_item_from_box(box)
            x1, y1, x2, y2 = coords
            
            # Create transparent image for this item
            item_result = np.zeros((*self.original_image.shape[:2], 4), dtype=np.uint8)
            item_result[:, :, :3] = self.original_image
            item_result[:, :, 3] = item_mask
            
            # Save full image
            item_image = Image.fromarray(item_result, 'RGBA')
            output_path = Path(output_dir) / f"{input_name}_box_{i+1}_full.png"
            item_image.save(output_path)
            
            # Save cropped version too
            cropped_result = item_result[y1:y2, x1:x2]
            cropped_image = Image.fromarray(cropped_result, 'RGBA')
            cropped_path = Path(output_dir) / f"{input_name}_box_{i+1}_cropped.png"
            cropped_image.save(cropped_path)
            
            print(f"✅ Saved full: {output_path}")
            print(f"✅ Saved cropped: {cropped_path}")
        
        print(f"🎉 Saved {len(self.boxes)} items to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Bounding Box BiRefNet - Draw boxes to select items")
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', default='bbox_items', help='Output directory')
    
    args = parser.parse_args()
    
    print("📦 Bounding Box BiRefNet - Individual Item Selection")
    print("=" * 55)
    
    try:
        selector = BBoxBiRefNet()
        boxes = selector.select_with_boxes(args.input, args.output)
        
        print(f"\n🎉 Selection complete!")
        print(f"📦 Drew {len(boxes)} bounding boxes")
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