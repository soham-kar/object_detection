"""Check Modal checkpoint volume contents."""
import modal

app = modal.App("check-ckpt")
vol = modal.Volume.from_name("wrdnet-checkpoints", create_if_missing=False)

@app.function(image=modal.Image.debian_slim(), volumes={"/checkpoints": vol}, timeout=60)
def check():
    import os
    vol.reload()
    lines = []
    try:
        items = os.listdir("/checkpoints")
        lines.append(f"Top-level: {items}")
        for item in sorted(items):
            full = os.path.join("/checkpoints", item)
            if os.path.isdir(full):
                files = os.listdir(full)
                lines.append(f"  {item}/ ({len(files)} files): {sorted(files)}")
            else:
                lines.append(f"  {item} ({os.path.getsize(full)} bytes)")
    except Exception as e:
        lines.append(f"Error: {e}")
    return '\n'.join(lines)

if __name__ == '__main__':
    with app.run():
        result = check.remote()
    print(result)