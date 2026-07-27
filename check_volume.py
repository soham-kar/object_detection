"""Check what's on the Modal Volume."""
import modal

app = modal.App('check-volume')
vol = modal.Volume.from_name('wrdnet-data', create_if_missing=True)

@app.function(volumes={'/data': vol}, timeout=60)
def check():
    import os
    lines = []
    items = os.listdir('/data')
    lines.append(f'Volume has {len(items)} top-level items:')
    for item in sorted(items):
        full = os.path.join('/data', item)
        if os.path.isdir(full):
            count = sum(len(files) for _, _, files in os.walk(full))
            lines.append(f'  {item}/ ({count} files)')
        else:
            lines.append(f'  {item}')
    return '\n'.join(lines)

if __name__ == '__main__':
    with app.run():
        result = check.remote()
    print(result)