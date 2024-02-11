#!/bin/bash
set -e

IMAGE_NAME="docker.ops.iszn.cz/mlops/petr/train"
TAG="${1:-errant}"

# docker build --no-cache --tag "${IMAGE_NAME}:${TAG}" --progress plain .
docker build --tag "${IMAGE_NAME}:${TAG}" --progress plain .
docker push "${IMAGE_NAME}:${TAG}"

echo "Done, use ${IMAGE_NAME}:${TAG} as image name."