#!/bin/bash
# ============================================================
# setup.sh — Environment setup for gemini_comp
# Python 3.12.3
# ============================================================

set -e

echo "=== gemini_comp environment setup ==="

# ----------------------------------------------------------
# 1. System dependencies (Ubuntu/Debian)
# ----------------------------------------------------------
echo "[1/3] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    awscli

# ----------------------------------------------------------
# 2. Create and activate a virtual environment (optional but recommended)
# ----------------------------------------------------------
echo "[2/3] Setting up Python virtual environment..."

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created virtual environment at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ----------------------------------------------------------
# 3. Install Python dependencies
# ----------------------------------------------------------
echo "[3/3] Installing Python packages..."

pip install --upgrade pip

pip install \
    numpy==2.4.2 \
    pandas==3.0.0 \
    polars==1.38.1 \
    matplotlib==3.10.8 \
    pillow==12.1.0 \
    pillow-heif==1.2.0 \
    PyMuPDF==1.26.7 \
    tqdm==4.67.3 \
    lxml==6.0.2 \
    requests==2.31.0 \
    rich==13.7.1 \
    boto3 \
    s3fs \
    jupyter

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To configure AWS credentials:"
echo "  aws configure"
echo ""
echo "To start Jupyter:"
echo "  jupyter notebook"
