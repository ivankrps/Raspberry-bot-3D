import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Import the RD3D+ model and related modules
from model.rd3d_plus import RD3D_plus
import torchvision

class RGBDSimilarity:
    def __init__(self, device='cuda', threshold=0.95):
        """
        Initializes the RGBDSimilarity class by loading the RD3D+ model and setting up preprocessing.

        Args:
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load pre-trained ResNet50 model for the backbone
        resnet = torchvision.models.resnet50(pretrained=True)

        # Initialize RD3D_plus model
        self.model = RD3D_plus(32, resnet)

        # Load pre-trained weights for RD3D_plus
        model_weights_path = '/home/rodriguez/Documents/GitHub/habitat/habitat-lab/examples/model_path/RD3D_plus.pth'
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

        # Load the model weights
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((352, 352)),  # RD3D+ expects images resized to 352x352
            transforms.ToTensor(),
            # Normalize using ImageNet mean and std (for 3 channels)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.threshold = threshold

    def preprocess_image(self, rgb_np, depth_np):
        """
        Preprocesses RGB and depth NumPy arrays for feature extraction.

        Args:
            rgb_np (numpy.ndarray): RGB image as a NumPy array with shape (H, W, 3).
            depth_np (numpy.ndarray): Depth image as a NumPy array with shape (H, W, 1) or (H, W).

        Returns:
            torch.Tensor: Preprocessed RGB-D image tensor of shape [1, 3, 2, 352, 352].
        """
        # Convert depth image to 3 channels by replicating the single channel
        if depth_np.ndim == 2:
            depth_np = np.expand_dims(depth_np, axis=2)  # Shape: (H, W, 1)
        depth_np_3c = np.repeat(depth_np, 3, axis=2)  # Shape: (H, W, 3)

        # Convert NumPy arrays to PIL Images
        rgb_image = Image.fromarray(rgb_np.astype('uint8'), mode='RGB')
        depth_image = Image.fromarray(depth_np_3c.astype('uint8'), mode='RGB')

        # Apply transforms
        rgb_image = self.transform(rgb_image)  # Shape: [3, 352, 352]
        depth_image = self.transform(depth_image)  # Shape: [3, 352, 352]

        # Add batch and time dimensions
        rgb_image = rgb_image.unsqueeze(0).unsqueeze(2)  # Shape: [1, 3, 1, 352, 352]
        depth_image = depth_image.unsqueeze(0).unsqueeze(2)  # Shape: [1, 3, 1, 352, 352]

        # Concatenate along the time dimension
        rgbd_image = torch.cat([rgb_image, depth_image], dim=2)  # Shape: [1, 3, 2, 352, 352]

        return rgbd_image.to(self.device)

    def extract_features(self, rgbd_tensor):
        """
        Extracts x4 features from the preprocessed RGB-D tensor using the RD3D+ model.

        Args:
            rgbd_tensor (torch.Tensor): Preprocessed RGB-D tensor.

        Returns:
            torch.Tensor: Extracted x4 feature tensor.
        """
        features = {}

        # Define a hook function to capture x4 features
        def get_x4_features(module, input, output):
            features['x4'] = output.detach()

        # Register the hook on layer4 (corresponds to x4)
        hook_handle = self.model.resnet.layer4.register_forward_hook(get_x4_features)

        with torch.no_grad():
            # Forward pass
            _ = self.model(rgbd_tensor)

        # Remove the hook
        hook_handle.remove()

        # Extract x4 features
        x4_features = features['x4']  # Shape: [1, 2048, 2, 11, 11]

        # Average over the temporal dimension (RGB and depth)
        x4_features = x4_features.mean(dim=2)  # Shape: [1, 2048, 11, 11]

        # Flatten the features
        x4_flat = x4_features.view(1, -1)  # Shape: [1, 2048 * 11 * 11]

        return x4_flat
    
    def extract_x4_features(self, rgbd_tensor):
        """
        Extracts x4 features from the preprocessed RGB-D tensor using the RD3D+ model.

        Args:
            rgbd_tensor (torch.Tensor): Preprocessed RGB-D tensor.

        Returns:
            torch.Tensor: Extracted x4 feature tensor.
        """
        features = {}

        # Define a hook function to capture x4 features
        def get_x4_features(module, input, output):
            features['x4'] = output.detach()

        # Register the hook on layer4 (corresponds to x4)
        hook_handle = self.model.resnet.layer4.register_forward_hook(get_x4_features)

        with torch.no_grad():
            # Forward pass
            _ = self.model(rgbd_tensor)

        # Remove the hook
        hook_handle.remove()

        # Extract x4 features
        x4_features = features['x4']  # Shape: [1, 2048, 2, 11, 11]

        # Average over the temporal dimension (RGB and depth)
        x4_features = x4_features.mean(dim=2)  # Shape: [1, 2048, 11, 11]

        return x4_features

    def compute_similarity(self, features1, features2):
        """
        Computes cosine similarity between two feature vectors.

        Args:
            features1 (torch.Tensor): Feature tensor of the first image.
            features2 (torch.Tensor): Feature tensor of the second image.

        Returns:
            float: Cosine similarity value.
        """
        # Normalize the feature vectors
        features1_norm = F.normalize(features1, p=2, dim=1)
        features2_norm = F.normalize(features2, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.sum(features1_norm * features2_norm, dim=1)
        return similarity.item()

    def compute_image_similarity(self, rgb_np1, depth_np1, rgb_np2, depth_np2):
        """
        Computes similarity between two RGB-D images represented as NumPy arrays.

        Args:
            rgb_np1 (numpy.ndarray): RGB image of the first observation.
            depth_np1 (numpy.ndarray): Depth image of the first observation.
            rgb_np2 (numpy.ndarray): RGB image of the second observation.
            depth_np2 (numpy.ndarray): Depth image of the second observation.

        Returns:
            float: Cosine similarity value between the two images.
        """
        # Preprocess images
        rgbd_tensor1 = self.preprocess_image(rgb_np1, depth_np1)
        rgbd_tensor2 = self.preprocess_image(rgb_np2, depth_np2)

        # Extract features
        features1 = self.extract_features(rgbd_tensor1)
        features2 = self.extract_features(rgbd_tensor2)

        # Compute similarity
        similarity = self.compute_similarity(features1, features2)
        return similarity

    def select_key_images(self, rgbs_np, depths_np):
        """
        Selects key images from a sequence of RGB-D images based on similarity threshold.

        Args:
            rgbs_np (numpy.ndarray): Array of RGB images with shape (N, H, W, 3).
            depths_np (numpy.ndarray): Array of depth images with shape (N, H, W) or (N, H, W, 1).
            threshold (float): Similarity threshold to select key images.

        Returns:
            List[int]: Indices of selected key images.
        """
        num_images = rgbs_np.shape[0]
        key_indices = [0]  # Always select the first image

        # Preprocess and extract features for the first image
        current_rgb = rgbs_np[0]
        current_depth = depths_np[0]
        current_features = self.extract_features(
            self.preprocess_image(current_rgb, current_depth)
        )

        for i in range(1, num_images):
            rgb = rgbs_np[i]
            depth = depths_np[i]
            features = self.extract_features(
                self.preprocess_image(rgb, depth)
            )
            similarity = self.compute_similarity(current_features, features)
            if similarity < self.threshold:
                key_indices.append(i)
                current_features = features

        # Ensure the last image is included
        if key_indices[-1] != num_images - 1:
            key_indices.append(num_images - 1)

        return key_indices
    
    def compute_similarity_tensor(self, features1, features2):
        """
        Computes cosine similarity between two feature tensors.

        Args:
            features1 (torch.Tensor): Feature tensor of the first image. Shape: [C, H, W]
            features2 (torch.Tensor): Feature tensor of the second image. Shape: [C, H, W]

        Returns:
            float: Cosine similarity value.
        """
        # Flatten the feature tensors into 1D vectors
        features1_flat = features1.view(-1)  # Shape: [C * H * W]
        features2_flat = features2.view(-1)  # Shape: [C * H * W]

        # Normalize the feature vectors
        features1_norm = F.normalize(features1_flat, p=2, dim=0)
        features2_norm = F.normalize(features2_flat, p=2, dim=0)

        # Compute cosine similarity
        similarity = torch.dot(features1_norm, features2_norm)
        return similarity.item()


# Example usage:
# rgbd_similarity = RGBDSimilarity()
# similarity_score = rgbd_similarity.compute_image_similarity(rgb_np1, depth_np1, rgb_np2, depth_np2)
# key_indices = rgbd_similarity.select_key_images(rgbs_np, depths_np, threshold=0.95)
