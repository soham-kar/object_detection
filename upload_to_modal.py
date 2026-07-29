#!/usr/bin/env python3
"""
Download data from Google Drive to Modal Volume using rclone.

Usage:
    # 1. Get your rclone config:
    cat ~/.config/rclone/rclone.conf

    # 2. Paste it into RCLONE_CONF below

    # 3. Set the Drive folder path and run:
    python upload_to_modal.py
"""

import modal

app = modal.App("gdrive-upload")

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

# Paste your rclone.conf content here (from: cat ~/.config/rclone/rclone.conf)
RCLONE_CONF = os.environ.get("RCLONE_CONF", """
[gdrive]
type = drive
client_id = PLACEHOLDER
client_secret = PLACEHOLDER
scope = drive
token = PLACEHOLDER
""")

# Google Drive folder path (the folder containing your data)
# Example: "WRDNet/data" or just "data" depending on your Drive structure
DRIVE_FOLDER = "object_detection"  # Your data folder on Google Drive

# Modal volume name (create with: modal volume create wrdnet-data)
VOLUME_NAME = "wrdnet-data"

# Remote name from rclone config (e.g., "gdrive")
REMOTE_NAME = "gdrive"

# ─── MODAL SETUP ─────────────────────────────────────────────────────────────

DATA_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip")
    .run_commands(
        "curl https://rclone.org/install.sh | bash",
    )
    .pip_install("tqdm")
)


@app.function(image=image, volumes={"/data": DATA_VOLUME}, timeout=86400, memory=8192)
def download_from_gdrive(rclone_conf: str, remote_name: str, drive_folder: str):
    """
    Download data from Google Drive to Modal Volume using rclone.
    
    Args:
        rclone_conf: Full content of rclone.conf file
        remote_name: rclone remote name (e.g., "gdrive")
        drive_folder: Folder path in Google Drive (e.g., "data")
    """
    import os
    import subprocess
    
    # Write rclone config
    os.makedirs("/root/.config/rclone", exist_ok=True)
    with open("/root/.config/rclone/rclone.conf", "w") as f:
        f.write(rclone_conf)
    
    print(f"rclone config written to /root/.config/rclone/rclone.conf")
    
    # Test connection
    print(f"\nTesting rclone connection to {remote_name}:")
    result = subprocess.run(
        ["rclone", "lsd", f"{remote_name}:"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return
    
    # List the target folder
    print(f"\nContents of {remote_name}:{drive_folder}/")
    result = subprocess.run(
        ["rclone", "ls", f"{remote_name}:{drive_folder}/"],
        capture_output=True, text=True
    )
    print(result.stdout[:5000])  # First 5000 chars
    
    # Count files
    file_count = len([l for l in result.stdout.strip().split('\n') if l])
    print(f"\nTotal files: {file_count}")
    
    # Download
    print(f"\nDownloading {remote_name}:{drive_folder}/ → /data/")
    print("This may take a while for large datasets...")
    
    result = subprocess.run(
        [
            "rclone", "copy",
            f"{remote_name}:{drive_folder}/",
            "/data/",
            "--progress",
            "--transfers", "4",       # 4 parallel downloads
            "--checkers", "8",         # 8 parallel file checkers
            "--stats", "30s",          # Print stats every 30s
            "--retries", "5",          # Retry failed transfers
            "--low-level-retries", "10",
            "--drive-chunk-size", "64M",  # Larger chunks for faster transfer
        ],
        capture_output=True, text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"\nError: {result.stderr}")
        return
    
    # Verify
    print(f"\n✅ Download complete! Verifying...")
    items = os.listdir("/data")
    total_files = sum(len(files) for _, _, files in os.walk("/data"))
    total_size = sum(os.path.getsize(os.path.join(root, f)) 
                     for root, _, files in os.walk("/data") for f in files)
    
    print(f"  Top-level items: {len(items)}")
    for item in sorted(items):
        full = os.path.join("/data", item)
        if os.path.isdir(full):
            count = sum(len(files) for _, _, files in os.walk(full))
            print(f"    {item}/ ({count} files)")
        else:
            size = os.path.getsize(full) / 1e6
            print(f"    {item} ({size:.1f} MB)")
    
    print(f"\n  Total files: {total_files:,}")
    print(f"  Total size: {total_size / 1e9:.1f} GB")
    
    # Commit to volume
    DATA_VOLUME.commit()
    print(f"\n✅ Data committed to Modal Volume '{VOLUME_NAME}'")


@app.function(image=image, volumes={"/data": DATA_VOLUME}, timeout=3600, memory=4096)
def download_checkpoint(rclone_conf: str, remote_name: str, drive_folder: str, phase: str = "phase0"):
    """
    Download Phase 0 checkpoint from Google Drive to Modal Volume.
    
    Args:
        rclone_conf: Full content of rclone.conf file
        remote_name: rclone remote name
        drive_folder: Folder in Drive containing checkpoints
        phase: phase name (phase0 or phase1)
    """
    import os
    import subprocess
    
    os.makedirs("/root/.config/rclone", exist_ok=True)
    with open("/root/.config/rclone/rclone.conf", "w") as f:
        f.write(rclone_conf)
    
    ckpt_dir = f"/data/checkpoints/{phase}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"Downloading checkpoints from {remote_name}:{drive_folder}/ → {ckpt_dir}/")
    
    result = subprocess.run(
        [
            "rclone", "copy",
            f"{remote_name}:{drive_folder}/",
            ckpt_dir + "/",
            "--progress",
            "--transfers", "4",
        ],
        capture_output=True, text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return
    
    # List downloaded checkpoints
    files = os.listdir(ckpt_dir)
    print(f"\nDownloaded checkpoints: {files}")
    
    DATA_VOLUME.commit()
    print(f"✅ Checkpoints saved to Modal Volume")


@app.local_entrypoint()
def main(
    mode: str = "data",
    remote: str = "gdrive",
    folder: str = "object_detection",
):
    """
    Download data or checkpoints from Google Drive to Modal Volume.
    
    Usage:
        # Download dataset:
        modal run upload_to_modal.py --mode data --remote gdrive --folder object_detection
        
        # Download Phase 0 checkpoints:
        modal run upload_to_modal.py --mode checkpoint --remote gdrive --folder object_detection/experiments/checkpoints/phase0
    """
    if mode == "data":
        print(f"\nDownloading data from {remote}:{folder}/ to Modal Volume...")
        download_from_gdrive.remote(RCLONE_CONF, remote, folder)
    elif mode == "checkpoint":
        print(f"\nDownloading checkpoints from {remote}:{folder}/ ...")
        download_checkpoint.remote(RCLONE_CONF, remote, folder, "phase0")
    else:
        print(f"Unknown mode: {mode}. Use 'data' or 'checkpoint'.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import click
        main()
    else:
        print("Google Drive → Modal Volume Upload Script")
        print()
        print("Usage:")
        print("  python upload_to_modal.py --mode data --remote gdrive --folder data")
        print("  python upload_to_modal.py --mode checkpoint --remote gdrive --folder checkpoints/phase0")
        print()
        print("Before running:")
        print("  1. Edit RCLONE_CONF in this file with your rclone.conf content")
        print("  2. Set DRIVE_FOLDER to your Google Drive folder name")
        print("  3. Set REMOTE_NAME to your rclone remote name")
        print()
        print("To get rclone.conf:")
        print("  cat ~/.config/rclone/rclone.conf")