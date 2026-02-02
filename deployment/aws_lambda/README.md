# Chesapeake Land Cover Segmentation - AWS Lambda Deployment

Containerized deployment of the Chesapeake 7-class land cover segmentation model using AWS Lambda and Docker.

## Overview

This deployment packages a fine-tuned Clay foundation model for semantic segmentation of NAIP imagery into land cover classes. The model identifies:

- Water
- Tree Canopy
- Low Vegetation
- Barren Land
- Impervious Surfaces (Other)
- Impervious Surfaces (Roads)
- No Data

## Architecture

- **Base Image**: AWS Lambda Python 3.11 (ARM64/Graviton2)
- **Framework**: PyTorch + PyTorch Lightning
- **Model**: Clay v1.5 foundation model + Chesapeake segmentation head
- **API**: REST endpoint via API Gateway (when deployed to AWS)
- **Storage**: Model checkpoints stored in S3 and downloaded at runtime

## Local Deployment (Recommended)

### Prerequisites

- Docker Desktop with at least 8 GB RAM allocated
- AWS CLI configured (for S3 access)
- Python 3.11+ (for visualization script)

### Build Docker Image

```bash
# From project root
docker build -f deployment/aws_lambda/Dockerfile -t chesapeake-segmentation:latest .
```

### Run Container Locally

```bash
docker run -p 9000:8080 \
  -e MODEL_BUCKET=clay-model-checkpoints \
  -e CHESAPEAKE_CHECKPOINT_KEY=chesapeake-7class-segment_epoch-39_val-iou-0.8765.ckpt \
  -e CLAY_CHECKPOINT_KEY=clay-v1.5.ckpt \
  -e CHESAPEAKE_CHECKPOINT_PATH=/tmp/chesapeake_model.ckpt \
  -e CLAY_CHECKPOINT_PATH=/tmp/clay_model.ckpt \
  -e AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) \
  -e AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key) \
  -e AWS_DEFAULT_REGION=ca-central-1 \
  chesapeake-segmentation:latest
```

### Test Inference

```bash
# In another terminal
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d @test_event_real.json \
  -o response.json
```

### Visualize Results

```bash
python visualize_response.py response.json segmentation_output.png
open segmentation_output.png
```

## AWS Lambda Deployment

### Infrastructure as Code

The deployment uses AWS SAM (Serverless Application Model) for infrastructure management:

- **Lambda Function**: Containerized inference endpoint
- **API Gateway**: REST API with `/predict` endpoint
- **IAM Role**: S3 read access for model checkpoints
- **ECR Repository**: Container image storage

### Deploy to AWS

```bash
./deploy.sh
```

This script:
1. Creates ECR repository
2. Builds Docker image
3. Pushes to ECR
4. Deploys CloudFormation stack via SAM
5. Outputs API Gateway endpoint

### Known Limitations

⚠️ **Current deployment to AWS Lambda will fail due to memory constraints.**

**Issue**: The combined model checkpoints (6.1 GB on disk, ~3+ GB in RAM when loaded) exceed Lambda's maximum memory allocation of 3008 MB.

**Observed Behavior**:
- Checkpoints download successfully to `/tmp`
- Chesapeake model (1.3 GB) loads successfully
- Clay base model (4.8 GB) loading fails with `OSError: [Errno 14] Bad address` when memory is exhausted

**First invocation times out** after ~143 seconds with 100% memory utilization.

## API Specification

### Request Format

```json
{
  "image": "BASE64_ENCODED_IMAGE_DATA",
  "output_size": [256, 256]  // Optional
}
```

**Image Requirements**:
- Format: 4-band NAIP imagery (RGB + NIR)
- Encoding: Base64
- Expected band order: Red, Green, Blue, NIR

### Response Format

```json
{
  "statusCode": 200,
  "body": {
    "prediction": [[1, 1, 2, ...], ...],  // 2D array of class IDs
    "shape": [224, 224],
    "classes": {
      "1": "Water",
      "2": "Tree Canopy",
      ...
    }
  }
}
```

## File Structure

```
deployment/aws_lambda/
├── Dockerfile              # Lambda container definition
├── template.yaml           # SAM/CloudFormation template
├── deploy.sh              # Automated deployment script
├── lambda_handler.py      # Lambda function handler
├── requirements.txt       # Python dependencies
├── test_event_real.json   # Sample test event
├── test_local.py          # Local testing script
├── visualize_response.py  # Convert predictions to PNG
└── .env                   # Configuration (AWS region, S3 bucket)
```

## Environment Variables

**Required for local testing**:
- `MODEL_BUCKET`: S3 bucket containing model checkpoints
- `CHESAPEAKE_CHECKPOINT_KEY`: S3 key for Chesapeake model
- `CLAY_CHECKPOINT_KEY`: S3 key for Clay base model
- `AWS_ACCESS_KEY_ID`: AWS credentials
- `AWS_SECRET_ACCESS_KEY`: AWS credentials
- `AWS_DEFAULT_REGION`: AWS region (e.g., ca-central-1)

**Set at runtime**:
- `CHESAPEAKE_CHECKPOINT_PATH`: Local path to Chesapeake checkpoint
- `CLAY_CHECKPOINT_PATH`: Local path to Clay checkpoint
- `METADATA_PATH`: Path to model metadata YAML

## Performance Metrics

### Local Docker Deployment
- **Cold start**: ~180 seconds (includes S3 download)
- **Warm inference**: ~2-5 seconds per image
- **Memory usage**: ~3 GB
- **Image size**: 4.2 GB (without checkpoints), ~10 GB (with checkpoints baked in)

### AWS Lambda (Attempted)
- **Deployment**: ✅ Successful
- **API Gateway**: ✅ Endpoint created
- **Cold start**: ❌ Timeout (143s, memory exhausted)
- **Status**: Not viable without model optimization

## Development Notes

### Testing Locally

The local Docker deployment fully replicates the Lambda execution environment using the AWS Lambda Runtime Interface Emulator, ensuring development/production parity.

### Checkpoint Loading

Original implementation used PyTorch Lightning's `load_from_checkpoint()` which failed in Lambda due to CLI argument parsing. Resolved by manually loading checkpoints and constructing the model.

### Memory Management

Lambda's `/tmp` directory (only writable location) configured with 10 GB ephemeral storage for checkpoint downloads. 

## Acknowledgments

- Clay foundation model: [Clay-foundation](https://github.com/Clay-foundation/model)
- Chesapeake Land Cover dataset
- AWS Lambda container image support
