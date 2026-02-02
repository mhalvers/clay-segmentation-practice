# Use AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum install -y \
    gcc \
    gcc-c++ \
    make \
    cmake \
    git \
    && yum clean all

# Copy requirements first for better caching
COPY deployment/aws_lambda/requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy the claymodel package
COPY claymodel/ ${LAMBDA_TASK_ROOT}/claymodel/

# Copy the lambda handler
COPY deployment/aws_lambda/lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Copy configuration files
COPY configs/metadata.yaml ${LAMBDA_TASK_ROOT}/configs/

# Create directory for model checkpoints
RUN mkdir -p /opt/ml/model

# Note: Model checkpoints should be downloaded at runtime from S3 or baked into the image
# For baking into image (increases image size significantly):
# COPY checkpoints/segment/chesapeake-7class-segment_epoch-39_val-iou-0.8765.ckpt /opt/ml/model/chesapeake_model.ckpt
# COPY path/to/clay-v1.5.ckpt /opt/ml/model/clay_model.ckpt

# Set environment variables for model paths
ENV CHESAPEAKE_CHECKPOINT_PATH=/opt/ml/model/chesapeake_model.ckpt
ENV CLAY_CHECKPOINT_PATH=/opt/ml/model/clay_model.ckpt
ENV METADATA_PATH=${LAMBDA_TASK_ROOT}/configs/metadata.yaml

# Set the CMD to your handler
CMD ["lambda_handler.lambda_handler"]
