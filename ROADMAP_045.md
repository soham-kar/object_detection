# WRDNet → 0.45 mAP Implementation Roadmap

> **Goal:** Reach 0.45+ mAP@50 on Foggy Cityscapes for top-journal publication.
> **Current:** 0.31 mAP (Phase 0, 640×640, YOLOv11s, FSG bypassed).
> **SOTA reference:** CAST-YOLO 43.3%, WAMNet 51.9%.

**IMPORTANT:** Do each step, verify it works, THEN move to the next. Do NOT
combine steps — each change risks new instability, and you need to isolate
which change helps.

---

## Phase A: Core Contribution Fix (FSG must work)

### Step 1 — Fix FSG normalization (clamp → LayerNorm)
- [ ] In `src/models/fsg.py` and `src/models/dg_fsg.py`, replace the
      `torch.clamp(fused, -10, 10)` with a proper `nn.LayerNorm` on the fused
      features.
- [ ] Rationale: the ±10 clamp is a band-aid for the CDMSA magnitude explosion.
      LayerNorm normalizes to zero-mean/unit-variance, preserving relative
      ordering without hard clipping.
- [ ] **Verify:** Run a smoke test — fused features should be normalized, no NaN.
- [ ] **Checkpoint:** Does the FSG now produce meaningful α maps (not ~0.5)?

### Step 2 — Re-enable FSG in Phase 1 and verify it trains
- [ ] Run Phase 2 (domain adaptation) with the FSG enabled + LayerNorm.
- [ ] **Verify:** mAP should improve over the 0.31 Phase 0 baseline.
- [ ] **Checkpoint:** If FSG helps, this is your core contribution validated.

---

## Phase B: Novel Aspect-Ratio Contribution

### Step 3 — Implement 1024×512 (2:1) resolution
- [ ] Change `configs/default.yaml`: `input_size: [512, 1024]`,
      `input_size_detect: [512, 1024]`, `input_size_dehaze: [384, 768]`.
- [ ] Update `src/data/transforms.py` — `Resize` and transforms to accept (H, W) tuple.
- [ ] Update `src/models/dehazeformer.py` — `output_size` tuple.
- [ ] Update `src/models/depth_decoder.py` — `output_size` tuple.
- [ ] Update `src/models/wrnet.py` — pass tuples through.
- [ ] **CRITICAL:** Update `src/evaluation/evaluator.py` box normalization
      (x by 1024, y by 512) — this is the most error-prone change.
- [ ] Update `src/data/*.py` datasets — `input_size` tuple.
- [ ] Update `src/models/yolov11.py` — `img_size` handling.
- [ ] **Verify:** Run a smoke test — all shapes correct, mAP not mis-scaled.
- [ ] **Checkpoint:** Does 2:1 aspect ratio improve mAP over 640×640?

---

## Phase C: Backbone Upgrades

### Step 4 — Upgrade detection backbone YOLOv11s → YOLOv11m
- [ ] Change `yolo_variant: 's'` → `'m'` in config.
- [ ] Update `src/models/yolov11.py` to load `yolo11m.pt`.
- [ ] Update `fsg_channels` if backbone channels change.
- [ ] **Verify:** Model loads, trains, no NaN.
- [ ] **Checkpoint:** Does YOLOv11m improve mAP?

### Step 5 — Upgrade restoration backbone DehazeFormer-T → S
- [ ] Change `dehazeformer_variant: 'T'` → `'S'` in config.
- [ ] **Verify:** Better restoration features → better FSG fusion.
- [ ] **Checkpoint:** Does DehazeFormer-S improve mAP?

---

## Phase D: Attention Enhancement (optional, if still short of 0.45)

### Step 6 — Add attention to detection neck
- [ ] Add a lightweight attention module (CBAM or cross-attention) to the YOLO neck.
- [ ] Inspired by CAST-YOLO (cross-attention) and WAMNet (EHViT).
- [ ] **Verify:** mAP improvement.

---

## Final: Evaluation & Paper

### Step 7 — Full evaluation
- [ ] Run final evaluation on Foggy Cityscapes, ACDC, Foggy Driving, Foggy Zurich.
- [ ] Generate α vs. depth plot (the "money shot" for DG-FSG).
- [ ] Run ablation studies (E0-E14 from IMPLEMENTATION_PLAN.md).
- [ ] Update `paper/methodology.md` with final results.

---

## Expected mAP trajectory (estimates)

| Step | Cumulative mAP |
|------|----------------|
| Baseline (current) | 0.31 |
| + FSG LayerNorm + Phase 2 | 0.36-0.40 |
| + 1024×512 resolution | 0.39-0.43 |
| + YOLOv11m | 0.44-0.48 |
| + DehazeFormer-S | 0.46-0.50 |
| + Attention | 0.48-0.52 |

**Honest note:** These are estimates. A solid 0.40 with the FSG + aspect-ratio
contributions is already publishable. Don't block the paper on hitting exactly
0.45.
