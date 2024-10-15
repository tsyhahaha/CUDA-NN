#!/bin/bash

export SOURCEDIR=../src
export GPU_SOURCE_FILES=$(find "$SOURCEDIR" -name '*.cu')

echo "Using GPU source files: $GPU_SOURCE_FILES"

# 确保 GPU_SOURCE_FILES 不为空
if [ -z "$GPU_SOURCE_FILES" ]; then
    echo "No .cu files found in $SOURCEDIR"
    exit 1
fi

nvcc $GPU_SOURCE_FILES test-add.cu -o test
