import numpy as np
import os 
from modules.rgbd_similarity import RGBDSimilarity
import cv2


root_folder = '/home/rodriguez/Documents/GitHub/Raspberry-bot-3D/paths_14'
color_folder = root_folder + f'/color_path-2/'
depth_folder = root_folder + '/depth_path-2/' # estimated depth
output_color_folder = root_folder + '/key_color-2/'
output_depth_folder = root_folder + '/key_depth-2/'
color_images_filenames = sorted(os.listdir(color_folder))
depth_images_filenames = sorted(os.listdir(depth_folder))
print(f"Number of color images: {len(color_images_filenames)}")
print(f"Number of depth images: {len(depth_images_filenames)}")

rgbd_simmilarity = RGBDSimilarity(threshold=0.9)

# Load color and depth images into lists
rgbs = []
depths = []

for fname in color_images_filenames:
    img_path = os.path.join(color_folder, fname)
    # Load image using cv2 and convert from BGR to RGB
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image {fname}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgbs.append(img_rgb)

for fname in depth_images_filenames:
    img_path = os.path.join(depth_folder, fname)
    # Load the depth image; if it is stored as a single channel image, cv2.IMREAD_UNCHANGED will work fine.
    depth = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"Warning: Could not load depth image {fname}")
        continue
    # If the depth image has 3 channels, convert it to grayscale.
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    depths.append(depth)

# Convert lists to numpy arrays (assumes all images are of the same size)
rgbs_np = np.array(rgbs)
depths_np = np.array(depths)

# Use the RGBDSimilarity instance to select key images based on similarity
key_indices = rgbd_simmilarity.select_key_images(rgbs_np, depths_np)
print("Selected key indices:", key_indices)

# Ensure that the output directories exist
os.makedirs(output_color_folder, exist_ok=True)
os.makedirs(output_depth_folder, exist_ok=True)

# Save the selected key images to the output folders
for idx in key_indices:
    # Retrieve the corresponding filenames
    color_filename = color_images_filenames[idx]
    depth_filename = depth_images_filenames[idx]

    # Get the loaded images
    color_img = rgbs_np[idx]
    depth_img = depths_np[idx]

    # Convert color image back to BGR for saving with cv2
    color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    # Save the images
    cv2.imwrite(os.path.join(output_color_folder, color_filename), color_img_bgr)
    cv2.imwrite(os.path.join(output_depth_folder, depth_filename), depth_img)
