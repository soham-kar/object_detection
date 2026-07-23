#!/bin/bash
# Download datasets for WRDNet training

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "=== WRDNet Data Download Script ==="
echo "This script downloads and prepares datasets for WRDNet training."
echo ""

# Foggy Cityscapes
echo "[1/4] Foggy Cityscapes"
echo "    Foggy Cityscapes requires manual download from:"
echo "    https://www.cityscapes-dataset.com/downloads/"
echo "    Download: leftImg8bit_trainvaltest_foggy.zip"
echo "    Place extracted data in: $DATA_DIR/foggy_cityscapes/"
echo ""

# ACDC
echo "[2/4] ACDC (Adverse Conditions Dataset)"
echo "    Download from: https://acdc.vision.ee.ethz.ch/"
echo "    Place extracted data in: $DATA_DIR/acdc/"
echo ""

# DAWN
echo "[3/4] DAWN (Detection in Adverse Weather Nature)"
echo "    Download from: https://www.kaggle.com/datasets/"
echo "    Place extracted data in: $DATA_DIR/dawn/"
echo ""

# RESIDE (for DehazeFormer pretraining)
echo "[4/4] RESIDE-6K (optional, for restoration pretraining)"
echo "    Download from: https://sites.google.com/view/reside-dehaze-datasets"
echo "    Place extracted data in: $DATA_DIR/reside/"
echo ""

echo "=== Setup Complete ==="
echo "After downloading, verify structure with:"
echo "  python -c \"from src.data.dataset import build_dataloaders; ...\""
