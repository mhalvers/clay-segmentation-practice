#!/usr/bin/env python3
"""
Test script for Chesapeake Segmentation Lambda function.
Run this locally to test the Lambda handler before deploying.
"""

import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from PIL import Image

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deployment.aws_lambda.lambda_handler import lambda_handler

load_dotenv()




def create_test_image(size=(224, 224), bands=4):
    """Create a test NAIP-like image."""
    # Create random image data matching NAIP format (4 bands: R, G, B, NIR)
    img_array = np.random.randint(0, 255, size=(size[0], size[1], bands), dtype=np.uint8)
    
    # For 4-band images, save as TIFF (PNG doesn't support 4 channels properly)
    from io import BytesIO
    import tifffile
    buffered = BytesIO()
    # TIFF expects (H, W, C) format
    tifffile.imwrite(buffered, img_array, photometric='rgb')
    buffered.seek(0)
    return buffered.getvalue()


def image_to_base64(image_bytes):
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def test_lambda_local():
    """Test the Lambda function locally."""
    print("Creating test image...")
    test_image_bytes = create_test_image()
    image_b64 = image_to_base64(test_image_bytes)
    print(f"CHESAPEAKE_CHECKPOINT_PATH: {os.getenv('CHESAPEAKE_CHECKPOINT_PATH', 'Not set')}")
    print(f"CLAY_CHECKPOINT_PATH: {os.getenv('CLAY_CHECKPOINT_PATH', 'Not set')}")
    print(f"METADATA_PATH: {os.getenv('METADATA_PATH', 'Not set')}")
    event = {
        "body": {
            "image": image_b64,
            "output_size": [256, 256]
        }
    }
    
    print("Invoking Lambda handler...")
    print("Note: Make sure model checkpoints are available at the expected paths")
    print("Set environment variables if needed:")
    print("  export CHESAPEAKE_CHECKPOINT_PATH=/path/to/chesapeake_model.ckpt")
    print("  export CLAY_CHECKPOINT_PATH=/path/to/clay_model.ckpt")
    print("  export METADATA_PATH=/path/to/metadata.yaml")
    print()
    
    try:
        response = lambda_handler(event, None)
        
        print(f"Status Code: {response['statusCode']}")
        
        if response['statusCode'] == 200:
            body = json.loads(response['body'])
            print(f"Prediction shape: {body['shape']}")
            print(f"Classes: {list(body['classes'].keys())}")
            print("✓ Test passed!")
        else:
            print(f"Error: {response['body']}")
            print("✗ Test failed!")
    
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


def test_with_real_image(image_path):
    """Test with a real NAIP image or .npy chip file."""
    print(f"Loading image from {image_path}...")
    
    if image_path.endswith('.npy'):
        # Load numpy array directly
        import tifffile
        from io import BytesIO
        chip = np.load(image_path)
        print(f"Chip shape: {chip.shape}, dtype: {chip.dtype}")
        
        # Convert (C, H, W) to (H, W, C) for TIFF
        chip_hwc = np.transpose(chip, (1, 2, 0))
        
        # Save as TIFF bytes
        buffered = BytesIO()
        tifffile.imwrite(buffered, chip_hwc, photometric='rgb')
        buffered.seek(0)
        image_bytes = buffered.getvalue()
    else:
        # Load as regular image file
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    
    image_b64 = image_to_base64(image_bytes)
    
    event = {
        "body": {
            "image": image_b64,
            "output_size": [256, 256]
        }
    }
    
    print("Invoking Lambda handler...")
    response = lambda_handler(event, None)
    
    print(f"Status Code: {response['statusCode']}")
    if response['statusCode'] == 200:
        body = json.loads(response['body'])
        print(f"Prediction shape: {body['shape']}")
        
        # Get prediction array
        pred_array = np.array(body['prediction'], dtype=np.uint8)
        
        # Create RGB colorized image
        colors = {
            0: (0, 0, 255),        # Water - blue
            1: (34, 139, 34),      # Tree Canopy - forest green
            2: (154, 205, 50),     # Low Vegetation - yellow green
            3: (210, 180, 140),    # Barren Land - tan
            4: (169, 169, 169),    # Impervious (Other) - gray
            5: (105, 105, 105),    # Impervious (Road) - dark gray
            6: (0, 0, 0),          # No Data - black
        }
        
        rgb_image = np.zeros((*pred_array.shape, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            mask = pred_array == class_id
            rgb_image[mask] = color
        
        # Save colorized RGB image
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_prediction.png"
        Image.fromarray(rgb_image).save(output_path)
        print(f"Saved colorized prediction to {output_path}")
        
        # Also save raw class IDs as numpy array for analysis
        npy_path = Path(image_path).parent / f"{Path(image_path).stem}_prediction.npy"
        np.save(npy_path, pred_array)
        print(f"Saved raw prediction array to {npy_path}")
    else:
        print(f"Error: {response['body']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with provided image path
        test_with_real_image(sys.argv[1])
    else:
        # Test with synthetic image
        test_lambda_local()
