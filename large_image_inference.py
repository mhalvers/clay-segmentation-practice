"""
Large image inference for land cover classification using a sliding window approach.

This module handles:
- Loading large GeoTIFF images
- Breaking them into overlapping patches/chips
- Running inference on each patch
- Stitching predictions back together
- Saving the result as a GeoTIFF with proper georeferencing
"""

import numpy as np
import torch
import torch.nn.functional as F
import rioxarray as rxr
from pathlib import Path
from typing import Tuple, Optional
import yaml
from box import Box
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class ChipDataset(Dataset):
    """
    Dataset wrapper for image chips to enable parallel data loading.
    
    Args:
        chips: List of image chips (numpy arrays)
        normalizer: Function to normalize chips
    """
    
    def __init__(self, chips, normalizer):
        self.chips = chips
        self.normalizer = normalizer
    
    def __len__(self):
        return len(self.chips)
    
    def __getitem__(self, idx):
        chip = self.chips[idx]
        normalized = self.normalizer(chip)
        return normalized


class LargeImageSegmentor:
    """
    Handles segmentation of large images using a sliding window approach.
    
    Args:
        model: Trained ChesapeakeSegmentor model
        metadata: Metadata containing normalization parameters
        chip_size: Size of the sliding window (default: 256)
        stride: Stride for the sliding window (default: 128 for 50% overlap)
        batch_size: Number of chips to process at once (default: 16)
        device: Device to run inference on (default: 'cpu')
    """
    
    def __init__(
        self, 
        model,
        metadata: Box,
        platform: str = "naip",
        chip_size: int = 256,
        stride: Optional[int] = None,
        batch_size: int = 16,
        device: str = "cpu",
        num_workers: int = 4,
        use_parallel: bool = True
    ):
        self.model = model.to(device)
        self.model.eval()
        self.metadata = metadata
        self.platform = platform
        self.chip_size = chip_size
        self.stride = stride if stride is not None else chip_size // 2  # 50% overlap by default
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.use_parallel = use_parallel
        
        # Get normalization parameters
        self.mean = list(metadata[platform].bands.mean.values())
        self.std = list(metadata[platform].bands.std.values())
        
    def load_image(self, image_path: str) -> Tuple[np.ndarray, object]:
        """
        Load a GeoTIFF image using rioxarray.
        
        Args:
            image_path: Path to the GeoTIFF file
            
        Returns:
            Tuple of (image array, rioxarray DataArray with georeference info)
        """
        da = rxr.open_rasterio(image_path)
        
        # Convert to numpy array (channels, height, width)
        image = da.values.astype(np.float32)
        
        return image, da
    
    def normalize_chip(self, chip: np.ndarray) -> torch.Tensor:
        """
        Normalize a chip using the platform-specific mean and std.
        
        Args:
            chip: Numpy array of shape (channels, height, width)
            
        Returns:
            Normalized torch tensor
        """
        chip_tensor = torch.from_numpy(chip).float()
        
        # Normalize
        mean_tensor = torch.tensor(self.mean).view(-1, 1, 1)
        std_tensor = torch.tensor(self.std).view(-1, 1, 1)
        normalized = (chip_tensor - mean_tensor) / std_tensor
        
        return normalized
    
    def extract_chips(self, image: np.ndarray) -> Tuple[list, list]:
        """
        Extract overlapping chips from a large image using a sliding window.
        
        Args:
            image: Image array of shape (channels, height, width)
            
        Returns:
            Tuple of (list of chips, list of (row, col) positions)
        """
        channels, height, width = image.shape
        chips = []
        positions = []
        
        for row in range(0, height - self.chip_size + 1, self.stride):
            for col in range(0, width - self.chip_size + 1, self.stride):
                chip = image[:, row:row + self.chip_size, col:col + self.chip_size]
                chips.append(chip)
                positions.append((row, col))
        
        # Handle edges if image doesn't divide evenly
        # Add chips at the right edge
        if width % self.stride != 0:
            for row in range(0, height - self.chip_size + 1, self.stride):
                col = width - self.chip_size
                chip = image[:, row:row + self.chip_size, col:col + self.chip_size]
                chips.append(chip)
                positions.append((row, col))
        
        # Add chips at the bottom edge
        if height % self.stride != 0:
            for col in range(0, width - self.chip_size + 1, self.stride):
                row = height - self.chip_size
                chip = image[:, row:row + self.chip_size, col:col + self.chip_size]
                chips.append(chip)
                positions.append((row, col))
        
        # Bottom-right corner if both edges need handling
        if height % self.stride != 0 and width % self.stride != 0:
            row = height - self.chip_size
            col = width - self.chip_size
            chip = image[:, row:row + self.chip_size, col:col + self.chip_size]
            chips.append(chip)
            positions.append((row, col))
        
        return chips, positions
    
    def predict_chips(self, chips: list, return_probs: bool = False) -> np.ndarray:
        """
        Run inference on a list of chips with optional parallel processing.
        
        Args:
            chips: List of image chips
            return_probs: If True, return softmax probabilities instead of class predictions
            
        Returns:
            Array of predictions (or probabilities) for each chip
        """
        if self.use_parallel and self.num_workers > 1:
            return self._predict_chips_parallel(chips, return_probs)
        else:
            return self._predict_chips_sequential(chips, return_probs)
    
    def _predict_chips_sequential(self, chips: list, return_probs: bool = False) -> np.ndarray:
        """
        Sequential batch processing of chips (original implementation).
        
        Args:
            chips: List of image chips
            return_probs: If True, return softmax probabilities instead of class predictions
            
        Returns:
            Array of predictions (or probabilities) for each chip
        """
        all_predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(chips), self.batch_size), desc="Processing chips"):
            batch_chips = chips[i:i + self.batch_size]
            
            # Normalize and create batch
            normalized_chips = [self.normalize_chip(chip) for chip in batch_chips]
            batch_tensor = torch.stack(normalized_chips).to(self.device)
            
            # Create dummy time and latlon tensors
            batch_size = len(batch_chips)
            time_tensor = torch.zeros(batch_size, 4).to(self.device)
            latlon_tensor = torch.zeros(batch_size, 4).to(self.device)
            
            # Create datacube
            datacube = {
                "pixels": batch_tensor,
                "time": time_tensor,
                "latlon": latlon_tensor,
            }
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(datacube)
                
            # Interpolate to original size if needed
            if outputs.shape[-2:] != (self.chip_size, self.chip_size):
                outputs = F.interpolate(
                    outputs, 
                    size=(self.chip_size, self.chip_size), 
                    mode="bilinear", 
                    align_corners=False
                )
            
            # Get predictions or probabilities
            if return_probs:
                # Return softmax probabilities for stitching
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_predictions.append(probs)
            else:
                # Return class predictions
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_predictions.append(preds)
        
        return np.concatenate(all_predictions, axis=0)
    
    def _predict_chips_parallel(self, chips: list, return_probs: bool = False) -> np.ndarray:
        """
        Parallel processing of chips using DataLoader with multiple workers.
        
        Args:
            chips: List of image chips
            return_probs: If True, return softmax probabilities instead of class predictions
            
        Returns:
            Array of predictions (or probabilities) for each chip
        """
        # Create dataset and dataloader
        dataset = ChipDataset(chips, self.normalize_chip)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device != "cpu"
        )
        
        all_predictions = []
        
        # Process batches with progress bar
        with tqdm(total=len(dataloader), desc="Processing chips (parallel)") as pbar:
            for batch_tensor in dataloader:
                batch_tensor = batch_tensor.to(self.device)
                
                # Create dummy time and latlon tensors
                batch_size = batch_tensor.shape[0]
                time_tensor = torch.zeros(batch_size, 4).to(self.device)
                latlon_tensor = torch.zeros(batch_size, 4).to(self.device)
                
                # Create datacube
                datacube = {
                    "pixels": batch_tensor,
                    "time": time_tensor,
                    "latlon": latlon_tensor,
                }
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(datacube)
                    
                # Interpolate to original size if needed
                if outputs.shape[-2:] != (self.chip_size, self.chip_size):
                    outputs = F.interpolate(
                        outputs, 
                        size=(self.chip_size, self.chip_size), 
                        mode="bilinear", 
                        align_corners=False
                    )
                
                # Get predictions or probabilities
                if return_probs:
                    # Return softmax probabilities for stitching
                    probs = F.softmax(outputs, dim=1).cpu().numpy()
                    all_predictions.append(probs)
                else:
                    # Return class predictions
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_predictions.append(preds)
                
                pbar.update(1)
        
        return np.concatenate(all_predictions, axis=0)
    
    def stitch_predictions(
        self, 
        predictions: np.ndarray, 
        positions: list, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Stitch predictions back together with overlap handling.
        
        Uses probability averaging in overlapping regions.
        
        Args:
            predictions: Array of probabilities for each chip (num_chips, num_classes, H, W)
            positions: List of (row, col) positions for each chip
            image_shape: (height, width) of the original image
            
        Returns:
            Stitched prediction map
        """
        height, width = image_shape
        num_classes = predictions.shape[1]  # Get from predictions shape
        
        # Create output array and count array for averaging overlaps
        output = np.zeros((num_classes, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)
        
        for prob_map, (row, col) in zip(predictions, positions):
            # prob_map is already (num_classes, H, W)
            output[:, row:row + self.chip_size, col:col + self.chip_size] += prob_map
            count[row:row + self.chip_size, col:col + self.chip_size] += 1
        
        # Average overlapping regions
        count = np.maximum(count, 1)  # Avoid division by zero
        output = output / count[np.newaxis, :, :]
        
        # Get final prediction by taking argmax of averaged probabilities
        final_prediction = np.argmax(output, axis=0)
        
        return final_prediction
    
    def predict_large_image(
        self, 
        image_path: str, 
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Run segmentation on a large image.
        
        Args:
            image_path: Path to input GeoTIFF
            output_path: Optional path to save output GeoTIFF
            
        Returns:
            Prediction map as numpy array
        """
        print(f"Loading image from {image_path}...")
        image, geo_data = self.load_image(image_path)
        
        print(f"Image shape: {image.shape}")
        print(f"Extracting chips with size {self.chip_size} and stride {self.stride}...")
        chips, positions = self.extract_chips(image)
        
        print(f"Extracted {len(chips)} chips")
        print("Running inference...")
        # Get probabilities instead of hard predictions for better stitching
        predictions = self.predict_chips(chips, return_probs=True)
        
        print("Stitching predictions...")
        final_prediction = self.stitch_predictions(
            predictions, 
            positions, 
            (image.shape[1], image.shape[2])
        )
        
        if output_path:
            self.save_prediction(final_prediction, geo_data, output_path)
        
        return final_prediction
    
    def save_prediction(
        self, 
        prediction: np.ndarray, 
        geo_data, 
        output_path: str
    ):
        """
        Save prediction as a GeoTIFF with proper georeferencing.
        
        Args:
            prediction: Prediction array
            geo_data: Original rioxarray DataArray with georeference info
            output_path: Path to save the output
        """
        print(f"Saving prediction to {output_path}...")
        
        # Create a new DataArray with the prediction
        import xarray as xr
        
        # Use the spatial coordinates from the original data
        pred_da = xr.DataArray(
            prediction[np.newaxis, :, :],  # Add channel dimension
            coords={
                'band': [1],
                'y': geo_data.y,
                'x': geo_data.x,
            },
            dims=['band', 'y', 'x']
        )
        
        # Copy the CRS and transform
        pred_da.rio.write_crs(geo_data.rio.crs, inplace=True)
        pred_da.rio.write_transform(geo_data.rio.transform(), inplace=True)
        
        # Save as GeoTIFF
        pred_da.rio.to_raster(output_path, dtype='uint8')
        
        print(f"Prediction saved successfully!")


def main():
    """
    Example usage of the LargeImageSegmentor.
    """
    from claymodel.finetune.segment.chesapeake_model import ChesapeakeSegmentor
    
    # Paths
    CHESAPEAKE_CHECKPOINT = "checkpoints/segment/chesapeake-7class-segment_epoch-39_val-iou-0.8765.ckpt"
    CLAY_CHECKPOINT = "checkpoints/clay-v1.5.ckpt"
    METADATA_PATH = "configs/metadata.yaml"
    
    # Input/output
    INPUT_IMAGE = "path/to/your/large_image.tif"
    OUTPUT_IMAGE = "path/to/output/prediction.tif"
    
    # Load model
    print("Loading model...")
    model = ChesapeakeSegmentor.load_from_checkpoint(
        checkpoint_path=CHESAPEAKE_CHECKPOINT,
        ckpt_path=CLAY_CHECKPOINT,
    )
    
    # Load metadata
    with open(METADATA_PATH) as f:
        metadata = Box(yaml.safe_load(f))
    
    # Create segmentor
    segmentor = LargeImageSegmentor(
        model=model,
        metadata=metadata,
        platform="naip",
        chip_size=256,
        stride=128,  # 50% overlap
        batch_size=16,
        num_workers=4,  # Number of parallel workers for data loading
        use_parallel=True,  # Enable parallel processing
        device="cpu"  # Use "cuda" if you have GPU
    )
    
    # Run prediction
    prediction = segmentor.predict_large_image(INPUT_IMAGE, OUTPUT_IMAGE)
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique classes: {np.unique(prediction)}")


if __name__ == "__main__":
    main()
