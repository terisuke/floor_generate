#!/bin/bash

# Exit on error
set -e

# 1. 基本ツールインストール (macOS with Homebrew)
# Check if Homebrew is installed
if ! command -v brew &> /dev/null
then
    echo "Homebrew could not be found. Please install Homebrew first."
    echo "See: https://brew.sh/"
    exit 1
fi

echo "Installing basic tools via Homebrew..."
brew install python@3.11 git cmake pkg-config poppler tesseract
# FreeCAD Cask install might need to be conditional or handled carefully
# For now, assuming user handles FreeCAD app installation manually or has it.
# brew install --cask freecad # Commenting out as GUI app install might not be desired in script for all users
echo "Basic tools installed."

# 2. Python仮想環境
PYTHON_VERSION=3.11
VENV_NAME="floorplan_env"

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists. Skipping creation."
else
    echo "Creating Python $PYTHON_VERSION virtual environment: $VENV_NAME ..."
    python$PYTHON_VERSION -m venv $VENV_NAME
    echo "Virtual environment created."
fi

# Activate virtual environment (instructions for user)
echo "\nTo activate the virtual environment, run:"
echo "source $VENV_NAME/bin/activate"

# 3. Pythonライブラリインストール
# Ensure pip is available in the venv (should be by default)
echo "\nInstalling Python libraries from requirements.txt..."
echo "Please activate the virtual environment first if you haven't already: source $VENV_NAME/bin/activate"

# Create a requirements.txt if it doesn't exist, based on comprehensive_mvp_requirements.md
REQUIREMENTS_FILE="requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "requirements.txt not found. Creating one from the MVP document."
    cat << EOF > $REQUIREMENTS_FILE
torch==2.3.0
torchvision
torchaudio
diffusers==0.19.3
transformers==4.31.0
huggingface_hub==0.16.4
tokenizers==0.13.3
accelerate==0.25.0
peft==0.4.0
opencv-python==4.8.1.78
Pillow==10.1.0
svgwrite==1.4.3
svglib==1.5.1
shapely==2.0.2
reportlab==4.0.7
pytesseract==0.3.10
pdf2image==1.16.3
easyocr==1.7.0
ortools==9.8.3296
streamlit==1.28.0
pandas==2.1.3
numpy==1.24.4
# freecad # This is problematic as a pip install for the app. Assume system install for FreeCAD app.
# Python bindings for FreeCAD might be installed differently or bundled.
EOF
    echo "requirements.txt created. Please review it."
fi

echo "If the virtual environment is active, pip install will use its pip."
echo "Consider running: pip install -r $REQUIREMENTS_FILE after activating the venv."

# Reminder for FreeCAD
echo "\nIMPORTANT: FreeCAD application (version 0.22 or as per requirements) should be installed separately."
echo "The FreeCAD Python bindings might need specific setup depending on your FreeCAD installation method."

echo "\nSetup script finished. Please activate the virtual environment and install Python packages."                                        