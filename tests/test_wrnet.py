"""Quick test script for WRDNet end-to-end forward pass."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config
from src.models.wrnet import WRDNet

# Suppress ultralytics verbose output
os.environ['YOLO_VERBOSE'] = 'False'

config = load_config('configs/default.yaml')
model = WRDNet(config)
model.eval()

print(f'WRDNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    out = model(x)

print(f'Inference OK!')
print(f'  restored:   {out["restored"].shape}')
print(f'  detections: {out["detections"].shape}')
print(f'  keys: {list(out.keys())}')
