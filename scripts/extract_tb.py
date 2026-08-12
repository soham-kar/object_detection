"""Extract and merge TensorBoard metrics from a run directory."""
import glob
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_merged(path):
    files = sorted(glob.glob(path))
    merged = {}
    for f in files:
        ea = EventAccumulator(f)
        ea.Reload()
        for tag in ea.Tags()['scalars']:
            merged.setdefault(tag, [])
            for ev in ea.Scalars(tag):
                merged[tag].append((ev.step, ev.value))
    # sort by step
    for tag in merged:
        merged[tag].sort(key=lambda x: x[0])
    return merged

if __name__ == '__main__':
    path = sys.argv[1]
    merged = load_merged(path)
    for tag in sorted(merged.keys()):
        print(f"=== {tag} ===")
        for step, val in merged[tag]:
            print(f"  step {step}: {val:.4f}")
