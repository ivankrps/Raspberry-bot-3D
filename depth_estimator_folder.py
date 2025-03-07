#!/usr/bin/env python3

import sys
import os
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# Model configuration
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
encoder = 'vits'

# Load the model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(DEVICE).eval()

def infer_depth(image_path, output_path):
    """Read an image, infer its depth, normalize the depth map, and save it."""
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not read image from", image_path)
        return
    # Perform depth inference
    depth = model.infer_image(frame)
    # Normalize depth map for visualization
    depth_min = depth.min()
    depth_max = depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_norm = (depth_norm * 255).astype(np.uint8)
    # Save the depth map
    cv2.imwrite(output_path, depth_norm)
    print(f"Saved depth image to {output_path}")

def process_folder(input_folder, output_folder):
    """Process all image files in input_folder and save corresponding depth maps in output_folder."""
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            print(f"Processing {input_path} -> {output_path}")
            infer_depth(input_path, output_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 depth_estimator_folder.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_folder(input_folder, output_folder)

