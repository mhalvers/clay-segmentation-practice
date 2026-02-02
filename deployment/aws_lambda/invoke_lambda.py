#!/usr/bin/env python3
"""
Example client for invoking the Chesapeake Segmentation Lambda function.
"""

import base64
import json
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
from PIL import Image


class ChesapeakeSegmentationClient:
    """Client for Chesapeake segmentation Lambda function."""
    
    def __init__(self, function_name: str, region: str = "ca-central-1"):
        """
        Initialize the client.
        
        Args:
            function_name: Name of the Lambda function
            region: AWS region
        """
        self.lambda_client = boto3.client("lambda", region_name=region)
        self.function_name = function_name
    
    def predict(
        self,
        image_path: str,
        output_size: Optional[tuple] = (256, 256),
    ) -> np.ndarray:
        """
        Run segmentation prediction on an image.
        
        Args:
            image_path: Path to input image
            output_size: Desired output size (height, width)
        
        Returns:
            Numpy array with class predictions
        """
        # Load and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # Create request payload
        payload = {
            "body": {
                "image": image_b64,
                "output_size": list(output_size) if output_size else None,
            }
        }
        
        # Invoke Lambda
        response = self.lambda_client.invoke(
            FunctionName=self.function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
        
        # Parse response
        response_payload = json.loads(response["Payload"].read())
        
        if response_payload["statusCode"] != 200:
            raise Exception(f"Lambda error: {response_payload['body']}")
        
        body = json.loads(response_payload["body"])
        prediction = np.array(body["prediction"])
        
        return prediction
    
    def predict_batch(
        self,
        image_paths: list[str],
        output_size: Optional[tuple] = (256, 256),
    ) -> list[np.ndarray]:
        """
        Run predictions on multiple images.
        
        Args:
            image_paths: List of image paths
            output_size: Desired output size
        
        Returns:
            List of prediction arrays
        """
        predictions = []
        for image_path in image_paths:
            pred = self.predict(image_path, output_size)
            predictions.append(pred)
        return predictions
    
    def save_prediction(
        self,
        prediction: np.ndarray,
        output_path: str,
        colorize: bool = True,
    ):
        """
        Save prediction to file.
        
        Args:
            prediction: Prediction array
            output_path: Output file path
            colorize: Whether to apply color mapping
        """
        if colorize:
            # Apply Chesapeake color scheme
            colors = {
                0: (0, 0, 0),           # Background/No Data
                1: (0, 0, 255),         # Water
                2: (34, 139, 34),       # Tree Canopy
                3: (154, 205, 50),      # Low Vegetation
                4: (210, 180, 140),     # Barren Land
                5: (169, 169, 169),     # Impervious (Other)
                6: (105, 105, 105),     # Impervious (Road)
            }
            
            colored = np.zeros((*prediction.shape, 3), dtype=np.uint8)
            for class_id, color in colors.items():
                mask = prediction == class_id
                colored[mask] = color
            
            image = Image.fromarray(colored)
        else:
            image = Image.fromarray(prediction.astype(np.uint8))
        
        image.save(output_path)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python invoke_lambda.py <image_path> [function_name]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    function_name = sys.argv[2] if len(sys.argv) > 2 else "chesapeake-segmentation-inference"
    
    print(f"Invoking Lambda function: {function_name}")
    print(f"Input image: {image_path}")
    
    # Create client
    client = ChesapeakeSegmentationClient(function_name)
    
    # Run prediction
    print("Running prediction...")
    prediction = client.predict(image_path, output_size=(256, 256))
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique classes: {np.unique(prediction)}")
    
    # Save result
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_prediction.png"
    client.save_prediction(prediction, str(output_path), colorize=True)
    
    print(f"Saved prediction to: {output_path}")
