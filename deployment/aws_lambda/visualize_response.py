"""
Convert Lambda response JSON to PNG visualization.
"""

import json
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Chesapeake land cover class colors
CLASS_COLORS = {
    1: [0, 197, 255],      # Water - bright blue
    2: [38, 115, 0],       # Tree Canopy - dark green
    3: [163, 255, 115],    # Low Vegetation - light green
    4: [255, 170, 0],      # Barren Land - orange
    5: [156, 156, 156],    # Impervious (Other) - gray
    6: [0, 0, 0],          # Impervious (Road) - black
    15: [197, 0, 255],     # No Data - magenta
}

CLASS_NAMES = {
    1: "Water",
    2: "Tree Canopy",
    3: "Low Vegetation",
    4: "Barren Land",
    5: "Impervious (Other)",
    6: "Impervious (Road)",
    15: "No Data"
}


def visualize_segmentation(response_file, output_file):
    """
    Convert segmentation JSON response to colored PNG.
    
    Args:
        response_file: Path to JSON response file
        output_file: Path to output PNG file
    """
    # Load response
    with open(response_file, 'r') as f:
        response = json.load(f)
    
    # Check status
    if response.get('statusCode') != 200:
        print(f"Error in response: {response.get('body')}")
        return
    
    # Parse body
    body = json.loads(response['body'])
    prediction = np.array(body['prediction'])
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique classes: {np.unique(prediction)}")
    
    # Create RGB image
    h, w = prediction.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        mask = prediction == class_id
        rgb_image[mask] = color
    
    # Save as PNG
    img = Image.fromarray(rgb_image)
    img.save(output_file)
    print(f"Saved segmentation visualization to {output_file}")
    
    # Print class distribution
    print("\nClass distribution:")
    for class_id in np.unique(prediction):
        count = np.sum(prediction == class_id)
        percentage = 100 * count / prediction.size
        name = CLASS_NAMES.get(int(class_id), f"Unknown ({class_id})")
        print(f"  {name}: {count} pixels ({percentage:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_response.py <response.json> [output.png]")
        sys.exit(1)
    
    response_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "segmentation_output.png"
    
    visualize_segmentation(response_file, output_file)
