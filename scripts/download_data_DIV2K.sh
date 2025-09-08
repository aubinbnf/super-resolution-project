#!/bin/bash
# ==========================================================
# Script to download the DIV2K dataset (HR images)
# ==========================================================

# Stop on error
set -e

# Project root directory (if script launched from anywhere)
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DATA_DIR="$PROJECT_DIR/data/raw/DIV2K"

# Creation of files
mkdir -p "$DATA_DIR"

# Official URLs (train + valid)
BASE_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K"
TRAIN_HR_URL="$BASE_URL/DIV2K_train_HR.zip"
VALID_HR_URL="$BASE_URL/DIV2K_valid_HR.zip"

# Download (if files are missing)
echo "Download of DIV2K..."
if [ ! -f "$DATA_DIR/DIV2K_train_HR.zip" ]; then
    wget -O "$DATA_DIR/DIV2K_train_HR.zip" "$TRAIN_HR_URL"
else
    echo "Train HR already downloaded"
fi

if [ ! -f "$DATA_DIR/DIV2K_valid_HR.zip" ]; then
    wget -O "$DATA_DIR/DIV2K_valid_HR.zip" "$VALID_HR_URL"
else
    echo "Valid HR already downloaded"
fi

# Decompressing
echo "Decompressing..."
unzip -n "$DATA_DIR/DIV2K_train_HR.zip" -d "$DATA_DIR/"
unzip -n "$DATA_DIR/DIV2K_valid_HR.zip" -d "$DATA_DIR/"

echo "Dataset downloaded and extracted in $DATA_DIR"
