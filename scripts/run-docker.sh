#!/bin/bash

# location: ./scripts/run-docker.sh
set -e

# Script to securely run Docker containers

if [ $# -ne 1 ]; then
  echo "Usage: $0 <environment>"
  echo "Environments: development, staging, production"
  exit 1
fi

ENV=$1

# Validate environment
if [[ ! "$ENV" =~ ^(development|staging|production)$ ]]; then
  echo "Invalid environment. Must be one of: development, staging, production"
  exit 1
fi

CONTAINER_NAME="fastapi-langgraph-$ENV"
IMAGE_NAME="fastapi-langgraph-template:$ENV"

echo "Starting Docker container for $ENV environment"

# Check if container already exists
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
  echo "Container $CONTAINER_NAME already exists. Removing it..."
  docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
  docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi


# Create logs directory if it doesn't exist
mkdir -p ./logs

# Run the container
echo "Running container $CONTAINER_NAME from image $IMAGE_NAME"
docker run -d \
  -p 8001:8000 \
  -v ./logs:/app/logs \
  --name $CONTAINER_NAME \
  $IMAGE_NAME 

echo "Container $CONTAINER_NAME started successfully"
echo "API is available at http://localhost:8000"
echo "To view logs, run: make docker-logs ENV=$ENV" 