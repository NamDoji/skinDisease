#!/bin/bash
# Download DermNet dataset from Kaggle
# Requires: kaggle CLI configured at ~/.kaggle/kaggle.json
# See: https://www.kaggle.com/docs/api

set -e

echo "Downloading DermNet dataset from Kaggle..."
kaggle datasets download -d shubhamgoel27/dermnet --path ./data/raw --unzip
echo "Download complete. Dataset available at: data/raw/"
echo ""
echo "Next step: python scripts/prepare_data.py"
