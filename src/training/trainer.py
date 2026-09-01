"""Main training loop for WRDNet."""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..models.wrnet import WRDNet
from ..utils.config import Config
from .losses import WRDNetLoss
from .optimizer import build_optimizer, build_scheduler
from ..domain_adaptation.fda import FDATransform


class WRDNetTrainer:
    """Trainer for WRDNet."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.model = WRDNet(config).to(self.device)

        # Loss — pass YOLO model for detection loss initialization
        # Use the underlying DetectionModel (nn.Module), not the YOLO wrapper
        yolo_model = self.model.yolo.model if hasattr(self.model.yolo, 'model') else None
        self.criterion = WRDNetLoss(config, yolo_model=yolo_model)

        # Optimizer and scheduler
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Logging
        self.log_dir = getattr(config, 'log_dir', 'experiments/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # Checkpointing
        self.checkpoint_dir = getattr(config, 'checkpoint_dir', 'experiments/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_interval = getattr(config, 'save_interval', 1)  # Save every epoch for preemption recovery

        # Early stopping — ENABLED on mAP@50.
        # NOTE: This was previously disabled with the comment "mAP stays near 0
        # for first 20+ epochs" — that was written for the 80-class COCO head,
        # which had mismatched class semantics and never learned. With the
        # 8-class Cityscapes head, mAP peaks early (~epoch 6) then OVERFITS
        # (mAP declines while val_loss keeps dropping). We must save best.pth
        # by mAP and stop when mAP stops improving to avoid overfitting.
        self.early_stopping = getattr(config, 'early_stopping', True)
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 10)
        self.early_stopping_metric = getattr(config, 'early_stopping_metric', 'mAP@50')
        self.best_metric = 0.0
        self.patience_counter = 0

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Mixed precision (AMP) — ENABLED with BF16 for A100 tensor-core speed.
        # A 12M-param model runs much faster in bf16 on A100 than FP32.
        # WHY BF16 NOT FP16: the FSG gate conv (Conv2d(512, 64, 3x3)) sums
        # ~4608 products of features ~8-10 magnitude → ~295k, which OVERFLOWS
        # FP16's max of 65504 → inf → BatchNorm (inf-inf)=NaN → alpha NaN →
        # fused NaN. BF16 has the SAME 8-bit exponent range as FP32 (max ~3e38),
        # so it cannot overflow. BF16 also needs NO GradScaler (no loss scaling).
        self.use_amp = True
        self.amp_dtype = torch.bfloat16
        self.scaler = None  # BF16 does not require gradient scaling
        if self.use_amp:
            print(f"  Mixed precision (AMP) enabled with {self.amp_dtype}")

        # FDA transform (input-level domain adaptation)
        self.use_fda = getattr(config, 'use_fda', False)
        self.fda_start_epoch = getattr(config, 'fda_start_epoch', 30)
        self.fda_transform = None
        if self.use_fda:
            self.fda_transform = FDATransform(beta=0.01)
            print(f"  FDA enabled (start epoch: {self.fda_start_epoch})")

    def _get_fda_beta(self, epoch: int) -> float:
        """Get FDA beta for current epoch from schedule."""
        schedule = getattr(self.config, 'fda_beta_schedule', None)
        if schedule is None:
            return 0.01  # Default beta

        beta = 0.0
        for epoch_threshold, b in schedule:
            if epoch >= epoch_threshold:
                if isinstance(b, list):
                    beta = b[0]  # Use lower bound of random range
                else:
                    beta = b
        return beta

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Main training loop.

        Args:
            train_loader: training data loader
            val_loader: validation data loader (optional)
        """
        epochs = getattr(self.config, 'epochs', 100)
        log_interval = getattr(self.config, 'log_interval', 100)

        # Apply Phase 0 freezing (freeze DehazeFormer to speed up T4 training)
        self._apply_phase_freeze()

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Update loss epoch for domain warmup
            self.criterion.set_epoch(epoch)

            # Train one epoch
            train_loss = self._train_epoch(train_loader, log_interval)

            # Validation
            if val_loader is not None:
                val_metrics = self._validate(val_loader)

                # Early stopping on mAP@50 (higher is better).
                # Save best.pth whenever mAP improves. This preserves the
                # best-generalizing model even if later epochs overfit.
                if self.early_stopping:
                    current_metric = val_metrics.get(self.early_stopping_metric, 0.0)
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.patience_counter = 0
                        self._save_checkpoint('best.pth')
                        print(f"  [BEST] New best {self.early_stopping_metric}={current_metric:.4f} at epoch {epoch}")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"Early stopping at epoch {epoch} (no mAP improvement for {self.patience_counter} epochs)")
                            break

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.early_stopping_metric, 0.0))
                else:
                    self.scheduler.step()

            # Save checkpoint every epoch (critical for Modal preemption recovery)
            self._save_checkpoint(f'epoch_{epoch+1}.pth')
            self._save_checkpoint('latest.pth')  # Always update latest
            
            # Commit to Modal volume if available (preemption recovery)
            try:
                import modal
                vol = modal.Volume.from_name('wrdnet-checkpoints', create_if_missing=False)
                vol.commit()
            except:
                pass  # Not on Modal, local training

        self.writer.close()

    def _apply_phase_freeze(self):
        """
        Freeze/unfreeze model components based on training phase.

        Phase 0 (warmup): Freeze DehazeFormer to speed up T4 training and
        prevent random YOLO head gradients from destroying pretrained weights.
        YOLO and FSG remain trainable.

        Phase 1 (DA): Unfreeze everything for fine-tuning on A100.
        """
        phase = getattr(self.config, 'phase', 'warmup')
        import torch.nn as nn

        if phase == 'warmup':
            # Freeze DehazeFormer
            frozen = 0
            for param in self.model.dehazeformer.parameters():
                param.requires_grad = False
                frozen += 1
            print(f"  Phase 0: Frozen DehazeFormer ({frozen} params) to speed up T4 training")
            # Ensure YOLO and FSG are trainable
            for param in self.model.yolo.parameters():
                param.requires_grad = True
            for param in self.model.fsg.parameters():
                param.requires_grad = True
        else:
            # Phase 1: unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True
            print("  Phase 1: All components trainable (DehazeFormer unfrozen)")

    def reset_fsg_gate(self):
        """
        Re-initialize the FSG gate to near-identity (alpha ≈ 0) so the YOLO
        head sees nearly the same features it was trained on in Phase 0.

        The FSG is bypassed in Phase 0 (use_fsg=False), so its gate weights
        are RANDOM when Phase 1 turns it on. Loading the Phase 0 checkpoint
        overwrites the model's fresh zero-init with those random weights,
        which produce spatially-varying alpha (0-1) → feature-distribution
        shift → box collapse → mAP 0.0000.

        This re-applies the near-identity init AFTER checkpoint load:
          - zero the final gate conv weights (logits ≈ bias everywhere)
          - set bias to -4.0 → Sigmoid(-4.0) ≈ 0.018 → fused ≈ 0.98*orig
        Gradients still flow through the zero weights, so the gate learns.
        """
        import torch.nn as nn
        if not hasattr(self.model, 'fsg') or self.model.fsg is None:
            print("  [reset_fsg_gate] No FSG found, skipping")
            return
        gates = self.model.fsg.gates
        for gate in gates:
            # Handle both gate structures:
            #   - Plain FSG: gate is an nn.Sequential (final Conv2d at [-2])
            #   - DG-FSG: gate is an nn.ModuleDict with a 'gating' key holding
            #     the nn.Sequential (final Conv2d at gating_net[-2])
            if isinstance(gate, nn.ModuleDict):
                gating_net = gate['gating']
            else:
                gating_net = gate
            # gating_net[-2] is the final Conv2d (before Sigmoid)
            nn.init.zeros_(gating_net[-2].weight)
            nn.init.constant_(gating_net[-2].bias, -4.0)
        print(f"  [reset_fsg_gate] Re-initialized {len(gates)} FSG gates to near-identity (alpha≈0.018)")

    def _move_to_device(self, batch: dict) -> dict:
        """Move batch dict to device, handling tensor lists (bboxes)."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                # List of tensors (e.g., bboxes) — move each to device
                moved[k] = [t.to(self.device) for t in v]
            elif isinstance(v, list):
                moved[k] = v  # List of non-tensors (e.g., paths)
            else:
                moved[k] = v
        return moved

    def _train_epoch(self, train_loader: DataLoader, log_interval: int) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Handle paired (synth, real) batches from PairedDADataset
            if 'synth' in batch and 'real' in batch:
                synth_batch = self._move_to_device(batch['synth'])
                real_batch = self._move_to_device(batch['real'])
                loss_batch = synth_batch  # Loss computed on synthetic (labeled)
            else:
                # Single dataset mode
                synth_batch = self._move_to_device(batch)
                real_batch = None
                loss_batch = synth_batch

            # Apply FDA (input-level domain adaptation) if enabled.
            # CRITICAL FIX: store the FDA-transformed image in a SEPARATE key
            # ('fda_image') instead of replacing 'image'. The original 'image'
            # must stay intact for YOLO detection — the detection labels (bboxes)
            # are for the ORIGINAL synthetic image. If FDA replaces 'image', the
            # labels no longer match the altered image → detection head confused
            # → mAP oscillates (0.40 → 0.31). FDA should only feed the
            # restoration branch (DehazeFormer), not the detection branch.
            if self.use_fda and self.fda_transform is not None:
                if self.current_epoch >= self.fda_start_epoch and real_batch is not None:
                    beta = self._get_fda_beta(self.current_epoch)
                    if beta > 0:
                        self.fda_transform.beta = beta
                        synth_batch['fda_image'] = self.fda_transform(
                            synth_batch['image'], real_batch['image']
                        )

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model.forward_train(synth_batch, real_batch)

                # Compute loss
                losses = self.criterion(outputs, loss_batch)
                loss = losses['total']

                # ── DFL diagnostic: check if x-bins collapse to bin 0 ──
                # The detection head's 'boxes' output is [B, reg_max*4, N] =
                # 16 DFL bins × 4 coords (x1, y1, x2, y2). If the x-coordinate
                # DFL bins collapse to bin 0 (while y stays spread), that's the
                # root cause of the x-collapse. Log the mean bin index per coord.
                # NOTE: Logged once per epoch (every 500 steps) to avoid noise.
                if self.global_step % 500 == 0 and 'detections_s' in outputs:
                    try:
                        det = outputs['detections_s']
                        if isinstance(det, (tuple, list)):
                            det = det[0]
                        if isinstance(det, dict) and 'boxes' in det:
                            boxes = det['boxes']  # [B, 64, N]
                            reg_max = 16
                            # Reshape to [B, 4, reg_max, N] and compute mean bin
                            b, c, n = boxes.shape
                            dist = boxes.view(b, 4, reg_max, n)  # [B, 4, 16, N]
                            probs = torch.softmax(dist, dim=2)
                            bins = torch.arange(reg_max, device=boxes.device).float()
                            mean_bin = (probs * bins.view(1, 1, reg_max, 1)).sum(dim=2)  # [B, 4, N]
                            mean_bin = mean_bin.mean(dim=(0, 2))  # [4] per coord
                            print(f"  [DFL] mean bin per coord (x1,y1,x2,y2): "
                                  f"[{mean_bin[0].item():.2f}, {mean_bin[1].item():.2f}, "
                                  f"{mean_bin[2].item():.2f}, {mean_bin[3].item():.2f}]")
                    except Exception as e:
                        pass  # Diagnostic only — never break training

                # NaN guard — if loss is NaN, SKIP this batch entirely.
                # Do NOT replace with a detached tensor: with FP16 AMP, GradScaler
                # breaks when backward() produces no gradients ("No inf checks
                # recorded"). Skipping the optimizer step is AMP-safe.
                skip_step = False
                if torch.isnan(loss) or torch.isinf(loss):
                    skip_step = True
                    print(f"\n  [NaN] Epoch {self.current_epoch+1} batch {batch_idx}:")
                    for k, v in losses.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: {v.item() if v.numel()==1 else v.shape}")
                    if 'detections_s' in outputs:
                        det = outputs['detections_s']
                        if isinstance(det, (tuple, list)):
                            det = det[0]
                        if isinstance(det, torch.Tensor):
                            print(f"    detections_s shape: {det.shape}, has_nan: {torch.isnan(det).any().item()}")
                        elif isinstance(det, dict):
                            print(f"    detections_s dict keys: {list(det.keys())}")
                            for dk, dv in det.items():
                                if isinstance(dv, torch.Tensor):
                                    print(f"      {dk}: shape={dv.shape}, has_nan={torch.isnan(dv).any().item()}")
                    if 'restored_s' in outputs:
                        print(f"    restored_s has_nan: {torch.isnan(outputs['restored_s']).any().item()}")

            # Backward pass. BF16 does NOT use GradScaler (no loss scaling needed).
            if not skip_step:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Logging
                total_loss += loss.item()
                self.global_step += 1

                if batch_idx % log_interval == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    })
                    for key, value in losses.items():
                        if isinstance(value, torch.Tensor):
                            self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
            else:
                # NaN batch — skip optimizer step, don't log
                print(f"  [SKIP] Skipping optimizer step for NaN batch {batch_idx}")

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def _validate(self, val_loader: DataLoader) -> dict:
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0.0

        # Collect predictions and targets for mAP
        from ..evaluation.evaluator import WRDNetEvaluator
        evaluator = WRDNetEvaluator(self.model, device=str(self.device))

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle paired format if present
                if 'synth' in batch:
                    batch = self._move_to_device(batch['synth'])
                else:
                    batch = self._move_to_device(batch)

                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model.forward_train(batch)
                    losses = self.criterion(outputs, batch)
                total_loss += losses['total'].item()

        avg_loss = total_loss / len(val_loader)

        # Compute mAP
        try:
            det_metrics = evaluator.evaluate_detection(val_loader)
            mAP_50 = det_metrics.get('mAP@50', 0.0)
            mAP_5095 = det_metrics.get('mAP@50:95', 0.0)
        except Exception as e:
            print(f"  WARNING: mAP computation failed: {e}")
            mAP_50 = 0.0
            mAP_5095 = 0.0

        metrics = {
            'val_loss': avg_loss,
            'mAP@50': mAP_50,
            'mAP@50:95': mAP_5095,
        }

        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)

        print(f"  Val: loss={avg_loss:.4f}, mAP@50={mAP_50:.4f}, mAP@50:95={mAP_5095:.4f}")

        return metrics

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        # Save current_epoch + 1 so that on resume, the training loop starts at
        # the NEXT epoch (not re-runs the just-completed one). The checkpoint is
        # saved AFTER an epoch completes, so self.current_epoch is the completed
        # epoch. Saving current_epoch+1 makes `for epoch in range(current_epoch,
        # epochs)` start at the next epoch after preemption recovery.
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, reset_bn: bool = False, strict: bool = True):
        """Load model checkpoint.
        
        Args:
            path: path to checkpoint file
            reset_bn: if True, reset BatchNorm running stats after loading.
                      Use when switching batch sizes (e.g., T4 Phase 0 → A100 Phase 1).
            strict: if False, allow missing/extra keys (for architecture changes between phases).
        """
        checkpoint = torch.load(path, map_location=self.device)
        ckpt_sd = checkpoint['model_state_dict']

        # Compatibility check: verify the YOLO head class-count matches.
        # If the checkpoint was trained with a different number of classes
        # (e.g., old 80-class head vs new 8-class head), loading it would
        # corrupt the head with garbage weights → NaN. Refuse to load.
        try:
            # Find the detect head's class projection weight in the checkpoint
            head_key = None
            for k in ckpt_sd.keys():
                if 'model.23.cv3' in k and k.endswith('.weight'):
                    head_key = k
                    break
            if head_key is not None:
                ckpt_nc = ckpt_sd[head_key].shape[0]
                # Find the current model's corresponding head weight
                cur_head_key = None
                for k in self.model.state_dict().keys():
                    if 'model.23.cv3' in k and k.endswith('.weight'):
                        cur_head_key = k
                        break
                if cur_head_key is not None:
                    cur_nc = self.model.state_dict()[cur_head_key].shape[0]
                    if ckpt_nc != cur_nc:
                        print(f"  [WARNING] Checkpoint head has {ckpt_nc} classes but model has {cur_nc}. "
                              f"Refusing to load incompatible checkpoint (would corrupt head → NaN).")
                        return False
        except Exception as e:
            print(f"  [WARNING] Could not verify head compatibility: {e}")

        self.model.load_state_dict(ckpt_sd, strict=strict)
        # Only load optimizer/scheduler if not strict (architecture changed → fresh optimizer)
        if strict:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        print(f"Loaded checkpoint from {path}")
        
        if reset_bn:
            # Reset BatchNorm running stats for new batch size
            # running_mean → 0, running_var → 1 (identity normalization)
            # Stats will re-calibrate from new batch size in first ~20 batches
            import torch.nn as nn
            bn_count = 0
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    bn_count += 1
            print(f"Reset {bn_count} BatchNorm running stats (new batch size re-calibration)")
            # reset_bn=True is ONLY used for the fresh Phase 0 → Phase 1
            # transition (force_fresh_phase1). The Phase 0 checkpoint's epoch
            # (e.g., 31) must NOT carry over — a fresh Phase 1 run starts at
            # epoch 0. Otherwise the training loop starts at epoch 32.
            self.current_epoch = 0
            self.best_metric = 0.0
