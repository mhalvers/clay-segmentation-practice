#!/bin/bash

# Chesapeake Segmentation Model - AWS Lambda Deployment Script
# This script builds and deploys the containerized Lambda function

set -e  # Exit on error

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading configuration from .env file..."
    source .env
fi

# Configuration
AWS_REGION="${AWS_REGION:-ca-central-1}"
ECR_REPO_NAME="chesapeake-segmentation"
LAMBDA_FUNCTION_NAME="chesapeake-segmentation-inference"
MODEL_BUCKET="${MODEL_BUCKET:-my-model-checkpoints}"
STACK_NAME="chesapeake-segmentation-stack"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo_info "Starting deployment to AWS Lambda..."
echo_info "AWS Account: ${AWS_ACCOUNT_ID}"
echo_info "Region: ${AWS_REGION}"
echo_info "ECR Repository: ${ECR_URI}"

# Step 1: Create ECR repository if it doesn't exist
echo_info "Step 1: Checking ECR repository..."
if ! aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} 2>/dev/null; then
    echo_info "Creating ECR repository: ${ECR_REPO_NAME}"
    aws ecr create-repository \
        --repository-name ${ECR_REPO_NAME} \
        --region ${AWS_REGION} \
        --image-scanning-configuration scanOnPush=true
else
    echo_info "ECR repository already exists"
fi

# Step 2: Login to ECR
echo_info "Step 2: Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${ECR_URI}

# Step 3: Build Docker image
echo_info "Step 3: Building Docker image..."
cd ../..  # Go to model root directory
docker build \
    -f deployment/aws_lambda/Dockerfile \
    -t ${ECR_REPO_NAME}:latest \
    .

# Step 4: Tag image for ECR
echo_info "Step 4: Tagging Docker image..."
docker tag ${ECR_REPO_NAME}:latest ${ECR_URI}:latest

# Step 5: Push to ECR
echo_info "Step 5: Pushing image to ECR..."
docker push ${ECR_URI}:latest

# Step 6: Upload model checkpoints to S3 (if needed)
# echo_info "Step 6: Checking model checkpoints in S3..."
# CHESAPEAKE_CKPT="checkpoints/segment/chesapeake-7class-segment_epoch-39_val-iou-0.8765.ckpt"

# if [ -f "${CHESAPEAKE_CKPT}" ]; then
#     echo_info "Uploading Chesapeake checkpoint to S3..."
#     aws s3 cp "${CHESAPEAKE_CKPT}" \
#         "s3://${MODEL_BUCKET}/chesapeake-7class-segment_epoch-39_val-iou-0.8765.ckpt" \
#         --region ${AWS_REGION} || echo_warn "Failed to upload checkpoint. Make sure bucket exists."
# else
#     echo_warn "Chesapeake checkpoint not found at ${CHESAPEAKE_CKPT}"
# fi

# Note: You'll need to upload clay-v1.5.ckpt separately if not already in S3
echo_warn "Make sure to upload clay-v1.5.ckpt to s3://${MODEL_BUCKET}/clay-v1.5.ckpt"

# Step 7: Deploy using SAM (optional - alternative to manual Lambda creation)
echo_info "Step 7: Would you like to deploy using AWS SAM? (y/n)"
read -r DEPLOY_SAM

if [ "$DEPLOY_SAM" = "y" ]; then
    echo_info "Deploying with AWS SAM..."
    
    # Navigate to project root (2 levels up from current script location)
    # deployment/aws_lambda -> deployment -> model (project root)
    PROJECT_ROOT="$(pwd)"
    
    echo_info "Building with SAM from ${PROJECT_ROOT}..."
    sam build --template deployment/aws_lambda/template.yaml
    
    # Deploy
    echo_info "Deploying stack..."
    sam deploy \
        --template-file .aws-sam/build/template.yaml \
        --stack-name ${STACK_NAME} \
        --capabilities CAPABILITY_IAM \
        --region ${AWS_REGION} \
        --parameter-overrides \
            ModelBucketName=${MODEL_BUCKET} \
        --image-repository ${ECR_URI} \
        --resolve-s3
    
    echo_info "Getting stack outputs..."
    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${AWS_REGION} \
        --query 'Stacks[0].Outputs' \
        --output table
else
    echo_info "Skipping SAM deployment."
    echo_info "To create Lambda function manually, use:"
    echo_info "aws lambda create-function \\"
    echo_info "  --function-name ${LAMBDA_FUNCTION_NAME} \\"
    echo_info "  --package-type Image \\"
    echo_info "  --code ImageUri=${ECR_URI}:latest \\"
    echo_info "  --role <your-lambda-execution-role-arn> \\"
    echo_info "  --timeout 300 \\"
    echo_info "  --memory-size 10240 \\"
    echo_info "  --region ${AWS_REGION}"
fi

echo_info "Deployment complete!"
echo_info ""
echo_info "Next steps:"
echo_info "1. Test your Lambda function with a sample image"
echo_info "2. Monitor CloudWatch logs for any issues"
echo_info "3. Adjust memory/timeout settings if needed"
echo_info ""
echo_info "To test locally with Docker:"
echo_info "docker run -p 9000:8080 ${ECR_REPO_NAME}:latest"
echo_info ""
echo_info "Then invoke with:"
echo_info "curl -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' -d @test_event.json"
