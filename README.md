# Chesapeake Land Cover Segmentation

A land cover segmentation project using the [Clay Foundation Model](https://github.com/Clay-foundation/model) for Chesapeake Bay watershed analysis. This project demonstrates the segmentation capabilities of the Clay model through containerized deployment.

> **Note**: This project is purely for educational purposes and personal learning in geospatial ML and deployment strategies.

## Overview

This repository focuses on the **segmentation** module of the Clay Foundation Model, specifically trained on the Chesapeake Conservancy Land Cover dataset. It includes:

- Pre-trained segmentation model for 7-class land cover classification
- Containerized deployment using Docker and AWS Lambda
- Local inference testing and visualization tools

**Land Cover Classes:**
- Water
- Tree Canopy / Forest
- Low Vegetation / Field
- Barren Land
- Impervious (other)
- Impervious (road)
- No Data

## Documentation

- **Model Training**: See [segment.md](claymodel/finetune/segment/segment.md) for information about the segmentation model architecture and training process
- **Deployment**: See [deployment README](deployment/aws_lambda/README.md) for comprehensive deployment instructions including:
  - Local Docker deployment with Lambda Runtime Interface Emulator
  - AWS Lambda containerized deployment (with known limitations)
  - API usage and visualization examples

## Quick Start

### Local Inference with Docker

```bash
# Build the Docker image (from deployment/aws_lambda directory)
cd deployment/aws_lambda
docker build -t chesapeake-segmentation:latest -f Dockerfile ../../

# Run locally
docker run -p 9000:8080 chesapeake-segmentation:latest

# Make a prediction
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d @test_event.json > response.json

# Visualize the result
python visualize_response.py response.json
```

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd model
```

Set up the environment:

```bash
mamba env create --file environment.yml
mamba activate claymodel
```

## Attribution

This project is based on the [Clay Foundation Model](https://github.com/Clay-foundation/model) and uses the Clay v1.5 foundation model with a Chesapeake-specific segmentation head. The original Clay model is licensed under Apache 2.0.

**What's included from Clay:**
- Core model architecture (`claymodel/model.py`, `claymodel/factory.py`)
- Segmentation module (`claymodel/finetune/segment/`)
- Pre-trained Clay v1.5 checkpoint and Chesapeake segmentation weights

## License

This project is licensed under the [Apache 2.0 License](LICENSE), consistent with the original Clay Foundation Model.
