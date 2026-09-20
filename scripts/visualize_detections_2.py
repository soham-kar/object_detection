import modal
import torch
import cv2
import numpy as np
import os
import sys
import glob
from torchvision.ops import nms

app = modal.App("wrdnet-risk-viz")

DATA_VOLUME = modal.Volume.from_name("wrdnet-data", create_if_missing=True)
CHECKPOINT_VOLUME = modal.Volume.from_name("wrdnet-checkpoints", create_if_missing=True)

# We must clone the repo in the image so the 'src' module exists
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "opencv-python-headless", "numpy", 
        "pyyaml", "timm", "ultralytics"
    )
    .run_commands(
        "git clone https://github.com/IDKiro/DehazeFormer.git /tmp/DehazeFormer",
        "git clone https://github.com/soham-kar/object_detection.git /tmp/object_detection"
    )
)

@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": DATA_VOLUME, "/checkpoints": CHECKPOINT_VOLUME},
    timeout=3600
)
def generate_risk_image():
    # 1. Add project root AND DehazeFormer to path INSIDE the remote function
    sys.path.insert(0, '/tmp/object_detection')
    sys.path.insert(0, '/tmp/DehazeFormer')  # <--- THIS IS THE FIX
    
    from src.utils.config import load_config
    from src.models.wrnet import WRDNet

    print("Loading model and checkpoint...")
    config = load_config('/tmp/object_detection/configs/default.yaml')
    model = WRDNet(config).cuda().eval().half()
    ckpt = torch.load('/checkpoints/phase1/best.pth', map_location='cuda')
    model.load_state_dict(ckpt['model_state_dict'])
    print("Model loaded successfully.")

    # Find ACDC validation images
    acdc_dir = '/data/rgb_anon_trainvaltest/rgb_anon/fog/val'
    if not os.path.exists(acdc_dir):
        acdc_dir = '/data/acdc/val' 
        
    image_paths = sorted(glob.glob(os.path.join(acdc_dir, '*.png')))[:4]
    if len(image_paths) < 4:
        image_paths = sorted(glob.glob(os.path.join(acdc_dir, '**', '*.png'), recursive=True))[:4]
        
    print(f"Found {len(image_paths)} images for visualization.")
    results = []
    
    for img_path in image_paths:
        # 1. Load and preprocess image
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (1024, 512))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().cuda().half() / 255.0
        
        # 2. Run Inference (Using the correct forward_train API)
        with torch.no_grad():
            outputs = model.forward_train({'image': img_tensor}, None)
            
        # 3. Extract Detections
        preds = outputs['detections_s'] if 'detections_s' in outputs else outputs['detections']
        if isinstance(preds, (tuple, list)):
            preds = preds[0] # Remove batch dim -> [4+nc, 8400]
            
        # YOLOv8/v11 format: cx, cy, w, h, class_scores...
        boxes_cxcywh = preds[:4, :].T
        scores = preds[4:, :].max(dim=0)[0]
        
        # Filter by confidence (0.25) and convert to xyxy
        conf_mask = scores > 0.25
        boxes_cxcywh = boxes_cxcywh[conf_mask]
        scores = scores[conf_mask]
        
        print(f"  Found {len(boxes_cxcywh)} raw detections above 0.25 conf.")
        
        if len(boxes_cxcywh) == 0:
            results.append(img_resized)
            continue
            
        boxes_xyxy = torch.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        
        # Scale boxes to image size (1024x512)
        boxes_xyxy[:, [0, 2]] *= 1024
        boxes_xyxy[:, [1, 3]] *= 512
        
        # Apply Non-Maximum Suppression (NMS)
        keep = nms(boxes_xyxy, scores, iou_threshold=0.45)
        boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        
        print(f"  {len(boxes_xyxy)} boxes remained after NMS.")
        
        # 4. Extract Depth Map
        depth_pred = outputs.get('depth_640', outputs.get('depth_pred'))
        if depth_pred is not None:
            # Depth decoder outputs 0-1 (Sigmoid). Multiply by max_depth (80m) to get metric depth
            depth_map = depth_pred.squeeze().cpu().float().numpy() * 80.0
            if depth_map.shape != (512, 1024):
                depth_map = cv2.resize(depth_map, (1024, 512))
        else:
            depth_map = np.zeros((512, 1024))
            
                # 5. Draw Risk Boxes
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            
            # Safety check: Skip boxes with Infinity or NaN values
            if not np.isfinite([x1, y1, x2, y2]).all():
                continue
                
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Clamp coordinates to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(1023, x2), min(511, y2)
            
            # Get depth at the bottom-center of the bounding box
            cx = int((x1 + x2) / 2)
            cy = int(y2 - (y2 - y1) * 0.1) # Slightly above the bottom edge
            cy = max(0, min(511, cy))
            cx = max(0, min(1023, cx))
            
            obj_depth = depth_map[cy, cx]
            
            # Determine Risk Level based on depth
            if obj_depth < 15.0:
                color = (0, 0, 255)      # Red (High Risk)
                label = f"HIGH RISK ({obj_depth:.1f}m)"
            elif obj_depth < 30.0:
                color = (0, 255, 255)    # Yellow (Medium Risk)
                label = f"MED RISK ({obj_depth:.1f}m)"
            else:
                color = (0, 255, 0)      # Green (Low Risk)
                label = f"LOW RISK ({obj_depth:.1f}m)"
                
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_resized, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        results.append(img_resized)

    # 6. Create and Save 2x2 Montage
    top = np.hstack((results[0], results[1]))
    bottom = np.hstack((results[2], results[3]))
    montage = np.vstack((top, bottom))
    
    save_path = '/checkpoints/phase1/risk_assessment.png'
    cv2.imwrite(save_path, montage)
    CHECKPOINT_VOLUME.commit()
    print(f"Success! Risk assessment montage saved to {save_path}")

@app.local_entrypoint()
def main():
    generate_risk_image.remote()