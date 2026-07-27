"""Check Modal Volume for label directories."""
import modal

app = modal.App("check-labels")
vol = modal.Volume.from_name("wrdnet-data", create_if_missing=False)

@app.function(image=modal.Image.debian_slim(), volumes={"/data": vol}, timeout=120)
def check_labels():
    import os
    vol.reload()
    lines = []
    
    # Check cityscapes labels
    cs_labels = "/data/cityscapes/labels"
    if os.path.isdir(cs_labels):
        train_labels = os.path.join(cs_labels, "train")
        val_labels = os.path.join(cs_labels, "val")
        train_count = sum(len(files) for _, _, files in os.walk(train_labels)) if os.path.isdir(train_labels) else 0
        val_count = sum(len(files) for _, _, files in os.walk(val_labels)) if os.path.isdir(val_labels) else 0
        lines.append(f"cityscapes/labels/train: {train_count} files")
        lines.append(f"cityscapes/labels/val: {val_count} files")
    else:
        lines.append("WARNING: cityscapes/labels NOT FOUND!")
    
    # Check ACDC labels
    acdc_labels = "/data/acdc_labels"
    if os.path.isdir(acdc_labels):
        train_count = sum(len(files) for _, _, files in os.walk(os.path.join(acdc_labels, "train"))) if os.path.isdir(os.path.join(acdc_labels, "train")) else 0
        val_count = sum(len(files) for _, _, files in os.walk(os.path.join(acdc_labels, "val"))) if os.path.isdir(os.path.join(acdc_labels, "val")) else 0
        lines.append(f"acdc_labels/train: {train_count} files")
        lines.append(f"acdc_labels/val: {val_count} files")
    else:
        lines.append("WARNING: acdc_labels NOT FOUND!")
    
    # Check cityscapes subdirs
    cs = "/data/cityscapes"
    if os.path.isdir(cs):
        for item in sorted(os.listdir(cs)):
            full = os.path.join(cs, item)
            if os.path.isdir(full):
                count = sum(len(files) for _, _, files in os.walk(full))
                lines.append(f"  cityscapes/{item}/ ({count} files)")
    
    # Check Foggy_Driving structure
    fd = "/data/Foggy_Driving"
    if os.path.isdir(fd):
        for item in sorted(os.listdir(fd)):
            full = os.path.join(fd, item)
            if os.path.isdir(full):
                count = sum(len(files) for _, _, files in os.walk(full))
                lines.append(f"  Foggy_Driving/{item}/ ({count} files)")
    
    return '\n'.join(lines)

if __name__ == '__main__':
    with app.run():
        result = check_labels.remote()
    print(result)