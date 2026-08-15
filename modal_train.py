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
# Free tier: "T4" | Paid: "L4", "A10G", "A100-40GB", "A100-80GB", "L40S"
GPU_TYPE = "A100-80GB"  # Phase 1 on A100-80GB (2× memory → bs=12, 2× faster, no OOM)

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
    # Fix CUDA OOM from memory fragmentation. The DehazeFormer feature
    # interpolate at 1024×512 allocates large blocks; expandable_segments lets
    # PyTorch grow segments instead of failing on fragmentation. This is the
    # fix recommended by the OOM error itself.
    # YOLO_AUTOINSTALL=false disables ultralytics' auto-install of optional
    # deps (e.g. pi-heif) which otherwise loops on every worker spawn and
    # delays training start.
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "YOLO_AUTOINSTALL": "false",
    })
)


# ──────────────────────────────────────────────────────────────────────────────
# Data Upload Function (from local machine)
# ──────────────────────────────────────────────────────────────────────────────

@app.function(image=image, volumes={"/data": DATA_VOLUME}, timeout=14400, cpu=2, memory=8192)
def receive_data(files_data: list):
    """
    Receive data files from local machine and save to Modal Volume.
    Called by upload_local() which sends files in chunks.

    Args:
        files_data: list of (relative_path, file_bytes) tuples
    """
    import os

    for rel_path, file_bytes in files_data:
        full_path = os.path.join("/data", rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(file_bytes)

    DATA_VOLUME.commit()


def upload_local(local_data_path: str = "data"):
    """
    Upload local data to Modal Volume using tar streaming.
    Bundles all files into a single tar stream — much faster than
    individual file uploads (avoids per-file overhead).

    Usage:
        python modal_train.py upload
    """
    import os
    import time
    import tarfile
    import io

    if not os.path.exists(local_data_path):
        print(f"ERROR: {local_data_path} not found!")
        return

    # Count files for info
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(local_data_path):
        for fname in files:
            total_files += 1
            total_size += os.path.getsize(os.path.join(root, fname))

    print(f"Uploading {local_data_path} to Modal Volume 'wrdnet-data'...")
    print(f"  Files: {total_files:,}")
    print(f"  Size:  {total_size / 1e9:.1f} GB")
    print()
    print("  Method: tar streaming (bundles all files into one upload)")
    print("  DO NOT close this terminal until you see 'Upload complete!'")
    print()

    start_time = time.time()

    # Create a tar archive — use /tmp (200GB free on your system)
    tar_path = "/tmp/wrdnet_data.tar"

    print("  Creating tar archive (this reads 110 GB from disk)...")
    tar_start = time.time()

    with tarfile.open(tar_path, "w") as tar:
        tar.add(local_data_path, arcname=".")

    tar_time = time.time() - tar_start
    tar_size = os.path.getsize(tar_path)
    print(f"  Tar created: {tar_size / 1e9:.1f} GB in {tar_time:.1f}s")

    print("  Uploading to Modal Volume...")
    upload_start = time.time()

    with DATA_VOLUME.batch_upload() as batch:
        batch.put_file(tar_path, "wrdnet_data.tar")

    DATA_VOLUME.commit()

    upload_time = time.time() - upload_start

    # Clean up local tar
    os.remove(tar_path)

    # Now extract on Modal
    print("  Extracting tar on Modal Volume...")
    extract_start = time.time()

    @app.function(
        image=image,
        volumes={"/data": DATA_VOLUME},
        timeout=3600,
        cpu=2,
        memory=4096,
    )
    def extract_tar():
        import subprocess
        import os
        # Extract tar to volume root
        subprocess.run(["tar", "xf", "/data/wrdnet_data.tar", "-C", "/data/"], check=True)
        # Remove tar file to save space
        os.remove("/data/wrdnet_data.tar")
        DATA_VOLUME.commit()
        # List top-level contents
        items = os.listdir("/data")
        print(f"  Extracted {len(items)} top-level items:")
        for item in sorted(items):
            full = os.path.join("/data", item)
            if os.path.isdir(full):
                count = sum(len(files) for _, _, files in os.walk(full))
                print(f"    {item}/ ({count} files)")
            else:
                print(f"    {item}")

    extract_tar.remote()

    extract_time = time.time() - extract_start

    total_time = time.time() - start_time
    print(f"\nUpload complete!")
    print(f"  Tar creation: {tar_time:.1f}s")
    print(f"  Upload:       {upload_time:.1f}s ({tar_size / (upload_time + 1e-8) / 1e6:.1f} MB/s)")
    print(f"  Extraction:   {extract_time:.1f}s")
    print(f"  Total time:   {total_time / 60:.1f} minutes")
    print(f"  Saved to Modal Volume 'wrdnet-data'")
    print(f"\nYou can now run: modal run modal_train.py --phase phase0")


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
    timeout=86400,  # 24 hours max (Modal's limit) — allows 30 epochs on T4
    memory=32768,   # 32 GB RAM — the paired DA dataset (synth + real + FDA
                    # fda_image) uses ~22.7GB, exceeding the old 16GB request.
                    # 32GB gives headroom so Modal doesn't throttle/OOM.
    cpu=16,         # 16 cores for the 12 dataloader workers (num_workers=12).
                    # Without this, Modal defaults to ~2-5 cores and the workers
                    # compete for CPU → GPU starved → slow epochs.
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

    # Sync to latest code (handles force-pushed/rewritten history)
    subprocess.run(["git", "fetch", "origin"], cwd=REPO, check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO, check=True)

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
        # Use default.yaml for Phase 1 — it contains ALL the current settings
        # (1024×512 resolution, 8-class head, multi-density fog, 1:3 DA ratio,
        # debug flags disable_da_losses/force_fresh_phase1, correct LR 2e-4).
        # The old wrnet_s.yaml is STALE (640×640, DG-FSG, lr 5e-4, lambda_domain
        # 0.1) and caused the mAP-0.0000 crashes. Do NOT use it.
        config_file = "configs/default.yaml"
    else:
        config_file = "configs/default.yaml"

    print(f"\n{'='*60}")
    print(f"WRDNet Training on Modal")
    print(f"{'='*60}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Phase: {phase}")
    print(f"  Config: {config_file}")
    print(f"{'='*60}\n")

    # Load config
    config = load_config(config_file)

    # CRITICAL: set config.phase to the actual training phase. The trainer's
    # _apply_phase_freeze() reads config.phase to decide whether to freeze
    # DehazeFormer. The YAML has phase:'warmup' hardcoded, so without this
    # override Phase 1 would freeze DehazeFormer (the opposite of what it needs).
    config.phase = phase

    # Apply overrides
    if batch_size is not None:
        config.batch_size = batch_size
    if epochs is not None:
        config.epochs = epochs
    if lr is not None:
        config.lr = lr

    # GPU memory-aware batch size selection
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 8

    # Phase-specific settings
    if phase == "phase0":
        config.use_fda = False
        config.use_dct_align = False
        config.use_fsg_consistency = False
        # Disable depth in Phase 0 — the depth decoder diverges to NaN with a
        # randomly-initialized head. Depth is auxiliary; enable it in Phase 1.
        config.use_depth = False
        config.use_dg_fsg = False
        # FSG is DISABLED in Phase 0. The FSG's CDMSA produces fused features
        # with magnitude 37-62 (vs YOLO's ~8), which shatters the DFL in the
        # detection head → x-coordinates collapse to the left edge. Since
        # DehazeFormer is frozen in Phase 0, FSG has nothing useful to fuse.
        # Bypass FSG so YOLO trains on clean features (mag ~8). Re-enable in
        # Phase 1 with a magnitude clamp on the fused features.
        config.use_fsg = False
        if batch_size is None:
            # T4 (16GB): 2, L4 (24GB): 6, A100 (40GB): 24, A100 (80GB): 96
            # Phase 0 freezes DehazeFormer → only YOLO+FSG train → low memory.
            # NOTE: 1024×512 uses ~1.28x more VRAM than 640×640.
            # bs=24 uses ~14GB and is SAFE on the 40GB A100 (verified in smoke
            # test). bs=48 OOMs because the DehazeFormer feature interpolate
            # spikes to 13.5GB at 1024×512. Keep bs=24 for the 40GB card.
            # On the 80GB A100, bs=48 uses ~27GB. Increase to 96 to fill the
            # card to ~54GB (safe, ~26GB headroom) for faster training.
            if gpu_mem_gb <= 16:
                config.batch_size = 2   # T4 (with AMP)
            elif gpu_mem_gb < 24:
                config.batch_size = 4   # L4/A10G
            elif gpu_mem_gb < 70:
                config.batch_size = 24  # A100-40GB (1024×512, ~14GB, safe)
            else:
                config.batch_size = 96  # A100-80GB or L40S (Phase 0, ~54GB)
        if epochs is None:
            # Phase 0: 50 epochs (sweet spot). Starting from COCO pretrained
            # weights, the backbone already knows object shapes — it just needs
            # to adapt to fog + the 8-class head. With multi-density fog (3x
            # data) + augmentation, mAP climbs steadily but plateaus around
            # epoch 40-50. 100 epochs would only add ~1-2% mAP for 2x compute.
            # Early stopping (patience=10) stops it at the peak automatically.
            config.epochs = 50
        if lr is None:
            # The 8-class head is RANDOMLY initialized (not COCO pretrained).
            # A high LR (1e-3) makes the DFL box regression loss explode to NaN.
            # Use a low LR so the new head learns stably.
            config.lr = 1e-4
    elif phase == "phase1":
        # BINARY-SEARCH DEBUG: when disable_da_losses is true, turn OFF all DA
        # losses so Phase 1 runs as pure fine-tuning (FSG ON, DehazeFormer
        # unfrozen, no DA). This isolates whether the mAP-0.0000 crash is from
        # the DA losses vs the FSG/unfrozen-DehazeFormer themselves.
        disable_da = getattr(config, 'disable_da_losses', False)
        if disable_da:
            # All DA losses OFF (pure fine-tuning)
            config.use_fda = False
            config.use_dct_align = False
            config.use_fsg_consistency = False
            print("  [DEBUG] disable_da_losses=True → Phase 1 running as PURE fine-tuning (all DA losses OFF)")
        else:
            # DA losses ON — respect the individual flags from config so we can
            # test one DA loss at a time (binary search). Set use_fda/use_dct_align/
            # use_fsg_consistency in default.yaml to control which are active.
            config.use_fda = getattr(config, 'use_fda', True)
            config.use_dct_align = getattr(config, 'use_dct_align', True)
            config.use_fsg_consistency = getattr(config, 'use_fsg_consistency', True)
            print(f"  [DEBUG] DA losses: FDA={config.use_fda}, DCT={config.use_dct_align}, FSG-cons={config.use_fsg_consistency}")
        config.real_datasets = ["acdc"]  # ONLY ACDC for DA training. Zurich is for evaluation only!
        if batch_size is None:
            # 1:3 real:synth ratio — config.batch_size is TOTAL images per step.
            # The DataLoader uses batch_size//4 items (each item = 3 synth + 1
            # real), so 12 total = 9 synth + 3 real.
            #
            # NOTE: DehazeFormer is UNFROZEN in Phase 1, so it retains activations
            # for backward → much higher memory than Phase 0 (where it was frozen).
            # At 1024×512, bs=6 uses ~38GB on the 40GB A100 (OOM at the feature
            # interpolate). On the 80GB A100, bs=12 uses ~76GB (2× memory, 2×
            # faster, safe headroom). bs=16 would exceed 80GB — too risky.
            #
            # ⚠️ OOM TRAP: when DCT or FSG-consistency is ON, the real path runs
            # (full DehazeFormer + YOLO on real images) → memory roughly doubles.
            # FDA-only skips the real path (my fix), so it can use bs=12. But
            # DCT/FSG need a smaller batch to stay within 80GB.
            real_path_on = config.use_dct_align or config.use_fsg_consistency
            if gpu_mem_gb <= 16:
                config.batch_size = 2    # T4 (with AMP): 2 total = 1 synth + 1 real
            elif gpu_mem_gb < 24:
                config.batch_size = 4    # L4/A10G: 4 total = 3 synth + 1 real
            elif gpu_mem_gb < 70:
                config.batch_size = 4    # A100-40GB: 4 total = 3 synth + 1 real (~38GB)
            else:
                # A100-80GB/L40S
                if real_path_on:
                    config.batch_size = 6    # 6 total = 4 synth + 2 real (~50GB, safe with real path)
                else:
                    config.batch_size = 12   # 12 total = 9 synth + 3 real (~57GB, FDA-only skips real path)
        if epochs is None:
            # Phase 1: 120 epochs normally, but we only have ~15 credits left.
            # Cap at 8 epochs for the fast FDA test (~8 hrs on A100-80GB).
            config.epochs = 8
        if lr is None:
            # Phase 1 LR lowered from 5e-4 → 2e-4. The 5x spike from Phase 0's
            # 1e-4 caused confidence collapse (mAP 0.31→0.14, predictions
            # 1600→300) as the optimizer aggressively ripped weights to satisfy
            # DA losses, destroying the detection thresholds. A gentler LR lets
            # DA align features without shattering the detection head.
            config.lr = 2e-4

    # Checkpointing to Modal Volume
    ckpt_dir = f"/checkpoints/{phase}"
    log_dir = f"/checkpoints/{phase}/logs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    config.checkpoint_dir = ckpt_dir
    config.log_dir = log_dir

    # Increase dataloader workers to keep the A100 fed. With only 2 workers,
    # the GPU sits idle waiting for the CPU to load + augment images
    # (RandomScale, ColorJitter are CPU-bound). 8 workers keep the A100 busy,
    # cutting epoch time significantly. A100 has ample CPU power.
    # Bumped to 12 for the paired DA dataset (loads synth + real per item).
    config.num_workers = 12

    print(f"  GPU Memory: {gpu_mem_gb:.1f} GB")
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
    # Auto-resume: if checkpoint exists, always load it (preemption recovery)
    if phase == "phase1" and resume:
        # FORCE-FRESH: if force_fresh_phase1 is set, ignore any existing Phase 1
        # checkpoints and always load fresh from Phase 0 + reset the FSG gate.
        # This prevents the footgun where stale Phase 1 checkpoints (with random
        # FSG weights from before the near-identity fix) are resumed instead of
        # applying the FSG fix.
        force_fresh = getattr(config, 'force_fresh_phase1', False)
        if force_fresh:
            print("  [DEBUG] force_fresh_phase1=True → ignoring existing Phase 1 checkpoints, loading fresh from Phase 0")
            phase1_latest = None
        else:
            # First check if Phase 1 already has checkpoints (resume mid-Phase 1)
            phase1_latest = os.path.join(ckpt_dir, "latest.pth")
            if not os.path.exists(phase1_latest):
                try:
                    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pth")]
                    if ckpts:
                        ckpts.sort(key=lambda x: int(x.replace("epoch_", "").replace(".pth", "")))
                        phase1_latest = os.path.join(ckpt_dir, ckpts[-1])
                except:
                    pass
        
        if phase1_latest is not None and os.path.exists(phase1_latest):
            # Resume from existing Phase 1 checkpoint (mid-Phase 1 recovery)
            print(f"Auto-resuming Phase 1 from {phase1_latest}")
            trainer.load_checkpoint(phase1_latest)
        else:
            # No Phase 1 checkpoint — load Phase 0 and reset BN stats
            ckpt_path = os.path.join("/checkpoints/phase0", "best.pth")
            if not os.path.exists(ckpt_path):
                try:
                    ckpts = [f for f in os.listdir("/checkpoints/phase0") if f.startswith("epoch_") and f.endswith(".pth")]
                    if ckpts:
                        ckpts.sort(key=lambda x: int(x.replace("epoch_", "").replace(".pth", "")))
                        ckpt_path = os.path.join("/checkpoints/phase0", ckpts[-1])
                except:
                    pass
            if os.path.exists(ckpt_path):
                print(f"Loading Phase 0 checkpoint: {ckpt_path}")
                print("  → Resetting BatchNorm stats for new batch size (T4→A100)")
                trainer.load_checkpoint(ckpt_path, reset_bn=True, strict=False)
                # The Phase 0 checkpoint contains RANDOM FSG gate weights (FSG
                # was bypassed in Phase 0). Re-init the gate to near-identity
                # (alpha≈0) so the YOLO head sees nearly the same features it
                # was trained on → prevents the box-collapse / mAP 0.0000 shock.
                trainer.reset_fsg_gate()
            else:
                print("WARNING: No Phase 0 checkpoint found, starting from scratch.")
    else:
        # Phase 0 or same-phase resume: auto-resume from latest checkpoint
        latest_path = os.path.join(ckpt_dir, "latest.pth")
        if not os.path.exists(latest_path):
            # Try best.pth
            latest_path = os.path.join(ckpt_dir, "best.pth")
        if not os.path.exists(latest_path):
            # Try epoch_*.pth
            try:
                ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("epoch_")])
                if ckpts:
                    latest_path = os.path.join(ckpt_dir, ckpts[-1])
            except:
                pass
        
        if os.path.exists(latest_path):
            print(f"Auto-resuming from {latest_path}")
            loaded = trainer.load_checkpoint(latest_path)
            if loaded is False:
                print("  → Incompatible checkpoint (head class mismatch). Starting from scratch.")
        else:
            print("No checkpoint found, starting from scratch.")

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

    # Sync to latest code (handles force-pushed/rewritten history)
    subprocess.run(["git", "fetch", "origin"], cwd=REPO, check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO, check=True)

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

    # Sync to latest code (handles force-pushed/rewritten history)
    subprocess.run(["git", "fetch", "origin"], cwd=REPO, check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO, check=True)

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
def upload():
    """Upload local data to Modal Volume.

    Usage:
        python modal_train.py upload
    """
    upload_local(local_data_path="data")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        # python modal_train.py upload
        upload_local(local_data_path="data")
    else:
        print("WRDNet Modal Training Script")
        print()
        print("Commands:")
        print("  python modal_train.py upload                          # Upload data from local (once)")
        print("  modal run modal_train.py --phase phase0               # Phase 0 training")
        print("  modal run modal_train.py --phase phase1 --resume      # Phase 1 training")
        print("  modal run modal_train.py::eval --phase phase0         # Evaluate")
        print("  modal run modal_train.py::alpha_depth_plot --phase 1  # α vs depth plot")
        print()
        print("GPU: " + GPU_TYPE)
        print("Data volume: wrdnet-data")
        print("Checkpoint volume: wrdnet-checkpoints")