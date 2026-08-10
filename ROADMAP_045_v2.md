# WRDNet → 0.45 mAP Implementation Plan (v2 — corrected)

> **Goal:** Reach 0.45+ mAP@50 on Foggy Cityscapes for top-journal publication.
> **Current:** 0.31 mAP (Phase 0, 640×640, YOLOv11s, FSG bypassed).
> **SOTA reference:** CAST-YOLO 43.3%, WAMNet 51.9%.

**IMPORTANT:** Do each step, verify it works, THEN move to the next. Do NOT
combine steps — each change risks new instability.

---

## Step 1: FSG Normalization Fix (Core Contribution)

**NOTE:** The FSG already has `output_bns` (BatchNorm) applied BEFORE the clamp.
The clamp is a hard clip ON TOP of the BN. The fix is to remove the clamp and
keep the BN (smooth normalization instead of hard clipping).

- [ ] `src/models/fsg.py` — `output_bns` already exists. Remove the
      `torch.clamp(fused, -10, 10)` line, keep `fused = self.output_bns[i](fused)`.
- [ ] `src/models/dg_fsg.py` — same change.
- [ ] **Verify:** Smoke test — fused features normalized, no NaN, α maps not ~0.5.

---

## Step 2: Backbone Capacity Upgrades

**⚠️ CORRECTION:** The plan said `YOLO('yolov8m.yaml')` — that is YOLOv8, NOT
YOLOv11. Your code uses YOLOv11. Use `yolo11m.pt` instead.

- [ ] `src/models/yolov11.py` — change model name from `yolo11s.pt` to
      `yolo11m.pt`. Load COCO weights with `strict=False`.
- [ ] `src/models/dehazeformer.py` — change variant from `'T'` to `'S'`.
- [ ] Update `fsg_channels` if backbone channels change (YOLOv11m may differ).
- [ ] **Verify:** Model loads, trains, no NaN.

---

## Step 3: 2:1 Aspect Ratio Resolution (Novel Research Gap)

- [ ] `configs/default.yaml` — `input_size_detect: [512, 1024]`,
      `input_size_dehaze: [384, 768]` (Height, Width).
- [ ] `src/data/transforms.py` — `Resize((512, 1024))`.
- [ ] `src/models/wrnet.py` — ensure `F.interpolate` uses `size=f_rest.shape[2:]`
      (dynamic), not hardcoded 640.
- [ ] `src/evaluation/evaluator.py` — **CRITICAL:** box normalization divides
      x by 1024, y by 512 (not scalar 640).
- [ ] **Verify:** Smoke test — all shapes correct, mAP not mis-scaled.

---

## Step 4: Modal & Training Config

**⚠️ CAUTION:** A100-80GB may not be available on your Modal plan. If not,
keep A100-40GB and lower batch size.

- [ ] `modal_train.py` — `GPU_TYPE = "A100-80GB"` (if available).
- [ ] Verify Phase 0: `epochs=50`, `lr=1e-4`, `use_fsg=False`,
      `batch_size=24`, `num_workers=8`.

---

## Step 5: Wipe & Launch Phase 0

- [ ] Wipe: `modal volume rm wrdnet-checkpoints phase0` (and `phase1`).
- [ ] Launch: `modal run modal_train.py::main --phase phase0 --detach`.
- [ ] **Target:** mAP climbs past 0.35, potentially 0.40+.

---

## Step 6: Launch Phase 1 (Domain Adaptation)

- [ ] Verify Phase 1: `use_fsg=True`, `use_fda=True`, `use_dct_align=True`,
      `lr=2e-4`. If OOM, drop batch size to 12.
- [ ] Launch: `modal run modal_train.py::main --phase phase1 --resume --detach`.
- [ ] **Target:** mAP pushes past 0.40+ on ACDC val.

---

## Step 7: Evaluation & The "Money Shot"

- [ ] Final eval: `modal run modal_train.py::eval --phase phase1 --dataset driving`.
- [ ] α-vs-depth plot: `modal run modal_train.py::alpha_depth_plot --phase phase1`.

---

## Expected mAP trajectory (estimates)

| Step | Cumulative mAP |
|------|----------------|
| Baseline (current) | 0.31 |
| + FSG fix + Phase 2 | 0.36-0.40 |
| + 1024×512 resolution | 0.39-0.43 |
| + YOLOv11m | 0.44-0.48 |
| + DehazeFormer-S | 0.46-0.50 |

**Honest note:** A solid 0.40 with the FSG + aspect-ratio contributions is
already publishable. Don't block the paper on hitting exactly 0.45.
