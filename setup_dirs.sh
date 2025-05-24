#!/bin/bash

# Create directory structure for the floor plan generation project
echo "Creating project directory structure..."

# Data directories
mkdir -p data/raw_pdfs
mkdir -p data/extracted
mkdir -p data/normalized
mkdir -p data/training
mkdir -p data/validation

# Source code directories
mkdir -p src/preprocessing
mkdir -p src/training
mkdir -p src/inference
mkdir -p src/constraints
mkdir -p src/freecad_bridge
mkdir -p src/ui

# Model directories
mkdir -p models/lora_weights
mkdir -p models/checkpoints

# Output directories
mkdir -p outputs/generated
mkdir -p outputs/svg
mkdir -p outputs/dxf
mkdir -p outputs/freecad

# Test and script directories
mkdir -p tests
mkdir -p scripts

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/preprocessing/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
touch src/constraints/__init__.py
touch src/freecad_bridge/__init__.py
touch src/ui/__init__.py

echo "Directory structure created successfully!"
