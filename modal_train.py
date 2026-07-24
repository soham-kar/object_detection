#!/usr/bin/env python3
"""
WRDNet Training on Modal — Serverless GPU Training

Runs WRDNet training on Modal's serverless GPUs (A100 40GB recommended).
Data is loaded from a Modal Volume (persistent storage).

Usage:
    # First time: upload data to Modal Volume
    modal volume create wrdnet-data
    python modal_train.py upload

    # Run Phase 0 (warmup, 30 epochs)
    modal run modal_train.py::train --phase phase0

    # Run Phase 1 (domain adaptation, 90 epochs)
    modal run modal_train.py::train --phase phase1

    # Run specific ablation experiment
    modal run modal_train.py::train --config configs/ablations/e3_joint_fsg.yaml

    # Evaluate on Foggy Driving
    modal run modal_train.py::evaluate

Cost estimate (A100 40GB):
    Phase 0: ~2 hours × $1.50/hr = ~$3
    Phase 1: ~8 hours × $1.50/hr = ~$12
    Total:   ~$15 for all experiments
"""

import os
import sys
import subprocess
import modal
import click

# ──────────────────────────────────────────────────────────────────────────────
# Modal Configuration
# ──────────────────────────────────────────────────────────────────────────────

app = modal.App("wrdnet-training")

# GPU selection — change this to match your Modal plan
GPU_TYPE = "A100-40GB"  # Options: T4, L4, A10G, A100-40GB, A100-80GB, L40S

# Persistent volume for data and checkpoints
DATA_VOLUME = modal.Volume.from_name("wrdnet-data", create_if_missing=True)
CHECKPOINT_VOLUME = modal.Volume.from_name("wrdnet-checkpoints", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget", "curl")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "ultralytics>=8.3.0",
        "timm>=1.0.27",
        "opencv-python-headless>=4.8.0",
        "pycocotools>=2.0.7",
        "tensorboard>=2.14.0",
        "thop>=0.1.1",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "albumentations>=1.3.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "gdown>=4.7.0",  # Google Drive downloader
    )
    .run_commands(
        "git clone https://github.com/IDKiro/DehazeFormer.git /tmp/DehazeFormer",
        # DehazeFormer doesn't have setup.py — just add to PYTHONPATH at runtime
    )
    .run_commands(
        "git clone https://github.com/soham-kar/object_detection.git /tmp/object_detection",
    )
)


# ──────────────────────────────────────────────────────────────────────────────
# Google Drive Download Function
# ──────────────────────────────────────────────────────────────────────────────

# Your Google Drive folder ID (extracted from the sharing link)
# Link: https://drive.google.com/drive/folders/19j_Nd1vx1DxHKz0kFh3oUOXjlV4id42y
GDRIVE_FOLDER_ID = "19j_Nd1vx1DxHKz0kFh3oUOXjlV4id42y"

@app.function(image=image, volumes={"/data": DATA_VOLUME}, timeout=7200)
def download_from_gdrive():
    """
    Download all data from Google Drive to Modal Volume.
    Run this once before training.

    Your Google Drive folder must be shared as "Anyone with the link can view".

    Usage:
        modal run modal_train.py::download
    """
    import os
    import subprocess

    print("Downloading data from Google Drive to Modal Volume...")
    print(f"  Folder ID: {GDRIVE_FOLDER_ID}")
    print(f"  Destination: /data/")
    print()

    # Use gdown to download the entire folder
    result = subprocess.run([
        "gdown", "--folder", f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
        "-O", "/data",
        "--remaining-ok",
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("\nTrying alternative method (individual files)...")

        # List files and download individually
        result2 = subprocess.run([
            "gdown", "--folder", f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
            "-O", "/data", "--no-cookies",
        ], capture_output=True, text=True)
        print(result2.stdout)
        if result2.returncode != 0:
            print("STDERR:", result2.stderr)
            print("\nERROR: Could not download from Google Drive.")
            print("Make sure the folder is shared as 'Anyone with the link can view'.")
            return

    # Verify download
    if os.path.exists("/data"):
        items = os.listdir("/data")
        print(f"\nDownloaded {len(items)} items to /data:")
        for item in sorted(items):
            full = os.path.join("/data", item)
            if os.path.isdir(full):
                count = sum(len(files) for _, _, files in os.walk(full))
                print(f"  {item}/ ({count} files)")
            else:
                size = os.path.getsize(full) / 1e6
                print(f"  {item} ({size:.1f} MB)")

    # Commit the volume
    DATA_VOLUME.commit()
    print(f"\nData saved to Modal Volume 'wrdnet-data'")
    print("You can now run training with: modal run modal_train.py --phase phase0")


# ──────────────────────────────────────────────────────────────────────────────
# Training Function
# ──────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/data": DATA_VOLUME,
        "/checkpoints": CHECKPOINT_VOLUME,
    },
    timeout=36000,  # 10 hours max
    memory=16384,   # 16 GB RAM
)
def train(
    phase: str = "phase0",
    config_path: str = None,
    batch_size: int = None,
    epochs: int = None,
    lr: float = None,
    resume: bool = False,
):
    """
    Run WRDNet training on Modal GPU.

    Args:
        phase: 'phase0' (warmup) or 'phase1' (domain adaptation)
        config_path: custom config path (overrides phase)
        batch_size: override batch size
        epochs: override number of epochs
        lr: override learning rate
        resume: resume from last checkpoint
    """
    import sys
    import os
    import torch

    # Set up paths
    REPO = "/tmp/object_detection"
    sys.path.insert(0, REPO)
    sys.path.insert(0, "/tmp/DehazeFormer")  # DehazeFormer module
    os.chdir(REPO)

    # Pull latest code
    subprocess.run(["git", "pull"], cwd=REPO, check=True)

    # Link data volume to project data directory
    if os.path.exists("data") and not os.path.islink("data"):
        os.rename("data", "data_backup")
    if not os.path.exists("data"):
        os.symlink("/data", "data")

    # Link checkpoint volume
    os.makedirs("/checkpoints", exist_ok=True)

    # Import WRDNet modules
    from src.utils.config import load_config
    from src.training.trainer import WRDNetTrainer
    from src.data.dataset import build_dataloaders

    # Determine config
    if config_path is not None:
        config_file = config_path
    elif phase == "phase0":
        config_file = "configs/default.yaml"
    elif phase == "phase1":
        config_file = "configs/wrnet_s.yaml"
    else:
        config_file = "configs/default.yaml"

    print(f"\n{'='*60}")
    print(f"WRDNet Training on Modal")
    print(f"{'='*60}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print(f"  Phase: {phase}")
    print(f"  Config: {config_file}")
    print(f"{'='*60}\n")

    # Load config
    config = load_config(config_file)

    # Apply overrides
    if batch_size is not None:
        config.batch_size = batch_size
    if epochs is not None:
        config.epochs = epochs
    if lr is not None:
        config.lr = lr

    # Phase-specific settings
    if phase == "phase0":
        config.use_fda = False
        config.use_dct_align = False
        config.use_fsg_consistency = False
        if batch_size is None:
            config.batch_size = 12  # A100 can handle 12
        if epochs is None:
            config.epochs = 30
        if lr is None:
            config.lr = 1e-3
    elif phase == "phase1":
        config.use_fda = True
        config.use_dct_align = True
        config.use_fsg_consistency = True
        config.real_datasets = ["acdc", "zurich"]
        if batch_size is None:
            config.batch_size = 6  # DA uses 2x memory (paired)
        if epochs is None:
            config.epochs = 90
        if lr is None:
            config.lr = 5e-4

    # Checkpointing to Modal Volume
    ckpt_dir = f"/checkpoints/{phase}"
    log_dir = f"/checkpoints/{phase}/logs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    config.checkpoint_dir = ckpt_dir
    config.log_dir = log_dir

    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  DA: FDA={getattr(config, 'use_fda', False)}, "
          f"DCT={getattr(config, 'use_dct_align', False)}, "
          f"FSG={getattr(config, 'use_fsg_consistency', False)}")
    print()

    # Build dataloaders
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config)
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")

    # Create trainer
    print("Creating trainer...")
    trainer = WRDNetTrainer(config)

    # Resume from checkpoint if requested
    if resume:
        ckpt_path = os.path.join(ckpt_dir, "best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{config.epochs}.pth")
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            trainer.load_checkpoint(ckpt_path)
        else:
            print("WARNING: No checkpoint found, starting from scratch.")

    # Train
    print(f"\nStarting {phase} training...")
    trainer.train(train_loader, val_loader)

    # Commit checkpoints to volume
    CHECKPOINT_VOLUME.commit()
    print(f"\n{phase} training complete! Checkpoints saved to Modal Volume.")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Function
# ──────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/data": DATA_VOLUME,
        "/checkpoints": CHECKPOINT_VOLUME,
    },
    timeout=3600,
    memory=16384,
)
def evaluate(
    phase: str = "phase0",
    dataset: str = "driving",
    visualize: bool = False,
):
    """
    Evaluate trained WRDNet model.

    Args:
        phase: which phase checkpoints to use ('phase0' or 'phase1')
        dataset: 'driving' (Foggy Driving) or 'acdc' (ACDC val)
        visualize: generate alpha map visualizations
    """
    import sys
    import os
    import torch

    REPO = "/tmp/object_detection"
    sys.path.insert(0, REPO)
    sys.path.insert(0, "/tmp/DehazeFormer")  # DehazeFormer module
    os.chdir(REPO)

    subprocess.run(["git", "pull"], cwd=REPO, check=True)

    if os.path.exists("data") and not os.path.islink("data"):
        os.rename("data", "data_backup")
    if not os.path.exists("data"):
        os.symlink("/data", "data")

    from src.utils.config import load_config
    from src.models.wrnet import WRDNet
    from src.data.dataset import build_test_loader, build_dataloaders
    from src.evaluation.evaluator import WRDNetEvaluator

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = WRDNet(config).to(device)

    ckpt_dir = f"/checkpoints/{phase}"
    ckpt_path = os.path.join(ckpt_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        # Find latest checkpoint
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("epoch_")])
        if ckpts:
            ckpt_path = os.path.join(ckpt_dir, ckpts[-1])

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("ERROR: No checkpoint found!")
        return

    # Build evaluator
    evaluator = WRDNetEvaluator(model, device=str(device))

    # Evaluate on specified dataset
    if dataset == "driving":
        print("\nEvaluating on Foggy Driving (101 images)...")
        test_loader = build_test_loader(config)
    elif dataset == "acdc":
        print("\nEvaluating on ACDC validation (100 images)...")
        config.batch_size = 4
        _, val_loader = build_dataloaders(config)
        test_loader = val_loader
    else:
        print(f"Unknown dataset: {dataset}")
        return

    # Detection metrics
    print("\nComputing detection metrics...")
    det_metrics = evaluator.evaluate_detection(test_loader)
    print(f"\nDetection Results:")
    print(f"  mAP@50:    {det_metrics.get('mAP@50', 0.0):.4f}")
    print(f"  mAP@50:95: {det_metrics.get('mAP@50:95', 0.0):.4f}")

    # Restoration metrics (only for synthetic data with clear GT)
    if dataset == "cityscapes":
        print("\nComputing restoration metrics...")
        rest_metrics = evaluator.evaluate_restoration(test_loader, has_gt=True)
        print(f"\nRestoration Results:")
        print(f"  PSNR: {rest_metrics.get('PSNR', 0.0):.2f} dB")
        print(f"  SSIM: {rest_metrics.get('SSIM', 0.0):.4f}")

    # Speed
    print("\nMeasuring inference speed...")
    fps = evaluator.measure_speed()
    print(f"  FPS: {fps:.1f}")

    # Alpha visualization
    if visualize:
        print("\nGenerating alpha map visualizations...")
        vis_dir = f"/checkpoints/{phase}/alpha_visualizations"
        evaluator.visualize_alpha_maps(test_loader, vis_dir, num_samples=20)
        CHECKPOINT_VOLUME.commit()
        print(f"Visualizations saved to {vis_dir}")

    print(f"\nEvaluation complete!")


# ──────────────────────────────────────────────────────────────────────────────
# Alpha vs Depth Plot Function
# ──────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/data": DATA_VOLUME,
        "/checkpoints": CHECKPOINT_VOLUME,
    },
    timeout=3600,
    memory=16384,
)
def plot_alpha_depth(phase: str = "phase1"):
    """
    Generate the α vs. depth correlation plot (the 'money shot' for DG-FSG).

    Args:
        phase: which phase checkpoints to use
    """
    import sys
    import os
    import torch

    REPO = "/tmp/object_detection"
    sys.path.insert(0, REPO)
    sys.path.insert(0, "/tmp/DehazeFormer")  # DehazeFormer module
    os.chdir(REPO)

    subprocess.run(["git", "pull"], cwd=REPO, check=True)

    if not os.path.exists("data"):
        os.symlink("/data", "data")

    from src.utils.config import load_config
    from src.models.wrnet import WRDNet
    from src.data.dataset import build_dataloaders

    config = load_config("configs/default.yaml")
    config.use_depth = True
    config.use_dg_fsg = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WRDNet(config).to(device)
    ckpt_path = f"/checkpoints/{phase}/best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # Get Cityscapes val loader (has depth GT)
    config.batch_size = 1
    config.fog_density = "0.02"
    _, val_loader = build_dataloaders(config)

    # Run the plot script
    import subprocess
    result = subprocess.run([
        sys.executable, "scripts/plot_alpha_vs_depth.py",
        "--checkpoint", ckpt_path,
        "--output", f"/checkpoints/{phase}/alpha_vs_depth.png",
        "--num-samples", "500",
    ], cwd=REPO, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    CHECKPOINT_VOLUME.commit()
    print(f"\nα vs depth plot saved to /checkpoints/{phase}/alpha_vs_depth.png")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Points
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    phase: str = "phase0",
    config: str = None,
    batch_size: int = None,
    epochs: int = None,
    lr: float = None,
    resume: bool = False,
):
    """Run WRDNet training on Modal GPU.

    Usage:
        modal run modal_train.py --phase phase0
        modal run modal_train.py --phase phase1 --resume
        modal run modal_train.py --config configs/ablations/e3_joint_fsg.yaml
    """
    train.remote(
        phase=phase,
        config_path=config,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        resume=resume,
    )


@app.local_entrypoint()
def eval(
    phase: str = "phase0",
    dataset: str = "driving",
    visualize: bool = False,
):
    """Evaluate trained model on Modal GPU.

    Usage:
        modal run modal_train.py::eval --phase phase0 --dataset driving
        modal run modal_train.py::eval --phase phase1 --dataset acdc --visualize
    """
    evaluate.remote(phase=phase, dataset=dataset, visualize=visualize)


@app.local_entrypoint()
def alpha_depth_plot(phase: str = "phase1"):
    """Generate α vs depth correlation plot.

    Usage:
        modal run modal_train.py::alpha_depth_plot --phase phase1
    """
    plot_alpha_depth.remote(phase=phase)


@app.local_entrypoint()
def download():
    """Download data from Google Drive to Modal Volume.

    Usage:
        modal run modal_train.py::download
    """
    download_from_gdrive.remote()


if __name__ == "__main__":
    print("WRDNet Modal Training Script")
    print()
    print("Commands:")
    print("  modal run modal_train.py::download                    # Download data from Google Drive (once)")
    print("  modal run modal_train.py --phase phase0               # Phase 0 training")
    print("  modal run modal_train.py --phase phase1 --resume      # Phase 1 training")
    print("  modal run modal_train.py::eval --phase phase0         # Evaluate")
    print("  modal run modal_train.py::alpha_depth_plot --phase 1  # α vs depth plot")
    print()
    print("GPU: " + GPU_TYPE)
    print("Data volume: wrdnet-data")
    print("Checkpoint volume: wrdnet-checkpoints")
    print(f"Google Drive folder: {GDRIVE_FOLDER_ID}")