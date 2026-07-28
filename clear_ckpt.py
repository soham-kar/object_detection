"""Clear Phase 0 checkpoints for fresh start."""
import modal

app = modal.App("clear-ckpt")
vol = modal.Volume.from_name("wrdnet-checkpoints", create_if_missing=False)

@app.function(image=modal.Image.debian_slim(), volumes={"/checkpoints": vol}, timeout=60)
def clear():
    import os
    import shutil
    vol.reload()
    lines = []
    phase0_dir = "/checkpoints/phase0"
    if os.path.exists(phase0_dir):
        files = os.listdir(phase0_dir)
        lines.append(f"Before: {files}")
        # Remove all .pth files
        for f in files:
            if f.endswith('.pth'):
                os.remove(os.path.join(phase0_dir, f))
                lines.append(f"  Deleted: {f}")
        # Keep logs directory
        remaining = os.listdir(phase0_dir)
        lines.append(f"After: {remaining}")
    else:
        lines.append("No phase0 directory found")
    vol.commit()
    return '\n'.join(lines)

if __name__ == '__main__':
    with app.run():
        result = clear.remote()
    print(result)