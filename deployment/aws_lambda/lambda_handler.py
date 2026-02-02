"""
AWS Lambda handler for Chesapeake segmentation model inference.
"""

import base64
import io
import json
import os
import traceback
from typing import Any, Dict

import boto3
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from claymodel.finetune.segment.chesapeake_model import ChesapeakeSegmentor

# Global variables for model caching
MODEL = None
DEVICE = torch.device("cpu")  # Lambda doesn't have GPU
s3_client = boto3.client('s3')


def download_model_from_s3():
    """Download model checkpoints from S3 if they don't exist locally."""
    model_bucket = os.environ.get("MODEL_BUCKET")
    chesapeake_key = os.environ.get("CHESAPEAKE_CHECKPOINT_KEY")
    clay_key = os.environ.get("CLAY_CHECKPOINT_KEY")
    
    chesapeake_path = "/tmp/chesapeake_model.ckpt"
    clay_path = "/tmp/clay_model.ckpt"
    
    # Download Chesapeake checkpoint
    if not os.path.exists(chesapeake_path):
        print(f"Downloading Chesapeake checkpoint from s3://{model_bucket}/{chesapeake_key}")
        s3_client.download_file(model_bucket, chesapeake_key, chesapeake_path)
    
    # Download Clay checkpoint
    if not os.path.exists(clay_path):
        print(f"Downloading Clay checkpoint from s3://{model_bucket}/{clay_key}")
        s3_client.download_file(model_bucket, clay_key, clay_path)


def load_model():
    """Load the model once and cache it."""
    global MODEL
    if MODEL is None:
        # Download models from S3 first
        download_model_from_s3()
        
        chesapeake_checkpoint_path = os.environ.get(
            "CHESAPEAKE_CHECKPOINT_PATH", "/tmp/chesapeake_model.ckpt"
        )
        clay_checkpoint_path = os.environ.get(
            "CLAY_CHECKPOINT_PATH", "/tmp/clay_model.ckpt"
        )
        metadata_path = os.environ.get(
            "METADATA_PATH", "/var/task/configs/metadata.yaml"
        )

        # Load checkpoint manually to avoid Lightning CLI issues
        print("Loading Chesapeake checkpoint...")
        checkpoint = torch.load(chesapeake_checkpoint_path, map_location=DEVICE)
        
        # Extract hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Create model instance with checkpoint parameters
        MODEL = ChesapeakeSegmentor(
            num_classes=hparams.get('num_classes', 7),
            ckpt_path=clay_checkpoint_path,
            lr=hparams.get('lr', 1e-4),
            wd=hparams.get('wd', 0.05),
            b1=hparams.get('b1', 0.9),
            b2=hparams.get('b2', 0.95),
        )
        
        # Load state dict
        MODEL.load_state_dict(checkpoint['state_dict'])
        MODEL.eval()
        MODEL.to(DEVICE)
        print("Model loaded successfully")
    return MODEL


def preprocess_image(image_data: bytes, normalize: bool = True) -> Dict[str, torch.Tensor]:
    """
    Preprocess the input image for model inference.
    
    Args:
        image_data: Raw image bytes (NAIP 4-band imagery expected)
        normalize: Whether to normalize the image
    
    Returns:
        Dictionary containing preprocessed datacube
    """
    # Load image
    image = Image.open(io.BytesIO(image_data))
    img_array = np.array(image)
    
    # Expected shape: (H, W, C) where C is 4 for NAIP (R, G, B, NIR)
    if img_array.ndim == 2:
        raise ValueError("Expected 4-band NAIP imagery, got grayscale")
    
    # Transpose to (C, H, W) and add batch dimension
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    # Normalize if needed (using NAIP statistics from metadata.yaml)
    # NAIP band order: red, green, blue, nir
    if normalize:
        means = torch.tensor([110.16, 115.41, 98.15, 139.04]).view(1, 4, 1, 1)
        stds = torch.tensor([47.23, 39.82, 35.43, 49.86]).view(1, 4, 1, 1)
        img_tensor = (img_tensor - means) / stds
    
    # Resize to 224x224 if necessary
    if img_tensor.shape[-2:] != (224, 224):
        img_tensor = F.interpolate(
            img_tensor, size=(224, 224), mode="bilinear", align_corners=False
        )
    
    # Create datacube with metadata
    # Note: time and latlon should be shape (batch, 4) to match training format
    batch_size = img_tensor.shape[0]
    datacube = {
        "pixels": img_tensor.to(DEVICE),
        "time": torch.zeros(batch_size, 4).to(DEVICE),  # Placeholder (batch, 4)
        "latlon": torch.zeros(batch_size, 4).to(DEVICE),  # Placeholder (batch, 4)
    }
    
    return datacube


def postprocess_output(outputs: torch.Tensor, output_size: tuple = None) -> np.ndarray:
    """
    Postprocess model outputs to get class predictions.
    
    Args:
        outputs: Model output logits
        output_size: Desired output size (H, W)
    
    Returns:
        Numpy array of class predictions
    """
    # Resize if needed
    if output_size:
        outputs = F.interpolate(
            outputs, size=output_size, mode="bilinear", align_corners=False
        )
    
    # Get class predictions
    preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
    return preds


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Expected event structure:
    {
        "body": {
            "image": "base64_encoded_image_data",
            "output_size": [256, 256]  # Optional
        }
    }
    
    Returns:
    {
        "statusCode": 200,
        "body": {
            "prediction": [[...]]  # 2D array of class predictions
        }
    }
    """
    try:
        # Parse input
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body", event)
        
        # Decode image
        image_b64 = body.get("image")
        if not image_b64:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No image data provided"})
            }
        
        image_data = base64.b64decode(image_b64)
        output_size = body.get("output_size")
        
        # Load model
        model = load_model()
        
        # Preprocess
        datacube = preprocess_image(image_data, normalize=True)
        
        # Run inference
        with torch.no_grad():
            outputs = model(datacube)
        
        # Postprocess
        predictions = postprocess_output(outputs, output_size)
        
        # Format response
        response = {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": predictions[0].tolist(),  # First batch item
                "shape": list(predictions[0].shape),
                "classes": {
                    "1": "Water",
                    "2": "Tree Canopy",
                    "3": "Low Vegetation",
                    "4": "Barren Land",
                    "5": "Impervious (Other)",
                    "6": "Impervious (Road)",
                    "15": "No Data"
                }
            })
        }
        
        return response
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {error_trace}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "type": type(e).__name__,
                "traceback": error_trace
            })
        }
