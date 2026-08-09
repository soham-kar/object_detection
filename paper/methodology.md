# Abstract

Object detection in autonomous driving degrades sharply under foggy
conditions, where the atmospheric scattering model couples scene radiance,
transmission, and depth in a way that confounds conventional detectors.
Existing approaches either dehaze the image as a preprocessing step and feed
the restored output to a detector, or train a detector directly on foggy
images without exploiting restoration cues. Both paradigms are suboptimal:
the former propagates restoration errors into detection, while the latter
ignores the physically grounded information that dehazing provides. We
propose the **Weather-Resilient Detection Unified Network (WRDNet)**, a
multi-branch architecture that jointly performs restoration, detection, and
depth estimation while adaptively fusing features across the restoration and
detection streams. At the core of WRDNet is a **Feature Selection Gate (FSG)**
that learns a per-pixel, multi-scale weighting between dehazed and original
features, allowing the network to exploit restoration cues where they are
beneficial without committing to a single restored image. We further introduce
a **Depth-Guided FSG (DG-FSG)** that uses estimated monocular depth to
actively modulate the fusion, exploiting the physical coupling between fog
severity and scene depth. To address the synthetic-to-real domain gap, we
propose a multi-level frequency-aware domain-adaptation strategy combining
Fourier Domain Adaptation, DCT-based feature alignment, and a novel
FSG-consistency loss. We also exploit the multi-density structure of the Foggy
Cityscapes dataset to train across three scattering coefficients, tripling the
effective data and enforcing fog-severity invariance. Extensive experiments on
Foggy Cityscapes, ACDC, Foggy Driving, and Foggy Zurich demonstrate that
WRDNet consistently outperforms sequential dehaze-then-detect baselines and
prior joint approaches, while the learned gating maps provide physically
interpretable evidence that the model adapts its fusion policy to scene depth
and fog density.

**Index Terms** — object detection, image dehazing, domain adaptation, feature
fusion, adverse weather, autonomous driving.

---

# I. Introduction

## A. Research Gaps and Contributions

A systematic review of the fog-removal and adverse-weather detection
literature reveals several persistent limitations that motivate our work. We
synthesize these gaps and state the corresponding contributions of WRDNet.

### A.1 Identified Research Gaps

**G1 — The sequential dehaze-then-detect paradigm is suboptimal.**
Classical prior-based methods, beginning with the Dark Channel Prior (DCP)
[1], and their lightweight variants [2, 3, 4], treat restoration as a
standalone preprocessing step that produces a single dehazed image for a
downstream detector. This decoupling is fundamentally limited: the
atmospheric scattering model couples restoration and detection through the
transmission map, yet the two stages are optimized independently. As a
consequence, restoration errors propagate directly into detection.
Over-aggressive dehazing can hallucinate artifacts in near-field regions
(where fog is thin and the prior is unreliable), while conservative
restoration leaves distant objects obscured. The detector is forced to
operate on a single, fixed restored representation that cannot adapt to the
spatially varying reliability of the dehazing output. Recent reviews of
defogging for object detection [5, 6] confirm that this sequential paradigm
consistently underperforms joint approaches, motivating feature-level
integration.

**G2 — Restoration quality is not aligned with detection objectives.**
Dehazing methods are predominantly optimized for image-quality metrics such
as PSNR, SSIM, BRISQUE, and NIQE [2, 7, 8, 9]. These perceptual metrics do
not correlate with downstream detection performance: an image that scores
well on structural similarity may still be suboptimal for object
localization, particularly for small and distant objects that are most
vulnerable under fog. The Cycle-Defog2Refog framework [7] and the Gamma-CNN
enhancement method [8] both optimize reconstruction fidelity without any
task-aware signal, while the histogram-equalization family [9, 10] applies
content-agnostic contrast enhancement. This metric misalignment means that
even a "perfectly" dehazed image, by perceptual standards, does not
guarantee improved detection, and can in fact degrade it by amplifying
noise.

**G3 — Single-density training limits fog-severity robustness.**
Foggy-driving benchmarks are typically constructed at a single scattering
coefficient $\beta$ [2, 4, 11], causing detectors to memorize a specific
atmospheric appearance. In real driving, visibility varies continuously
with fog density, and a detector trained at one operating point fails to
generalize across the continuum. The multi-density nature of the Foggy
Cityscapes dataset [12] — which renders each scene at
$\beta \in \{0.005, 0.01, 0.02\}$ — is rarely exploited, leaving a
significant source of physically grounded training signal unused.

**G4 — The synthetic-to-real domain gap is unaddressed.**
Methods trained exclusively on synthetic fog [2, 3, 7] degrade on real
foggy images due to the distribution shift between simulated and physical
fog. Prior work either ignores this gap entirely or relies on large,
manually curated real datasets that are expensive to obtain. The
Cycle-Defog2Refog framework [7] explicitly acknowledges that its
physics-based refogging model fails under dense fog, and that its real
dataset (MRFID) is too small (200 scenes) for robust generalization.
Domain-adaptation methods for adverse weather [13, 14] address this gap but
focus on a single alignment level, leaving multi-level alignment
unexplored.

**G5 — Depth information is underutilized.**
The atmospheric scattering model explicitly couples fog severity to scene
depth ($t(\mathbf{x}) = e^{-\beta d(\mathbf{x})}$), yet existing joint
dehazing-detection methods either ignore depth entirely or output it as a
parallel prediction that never feeds back into the detection decision.
Methods such as DEHRFormer and DCL estimate depth as a byproduct of
dehazing but do not use it to modulate the fusion of restored and original
features. This is a missed opportunity: depth is a physically meaningful
prior that indicates *where* dehazing is most beneficial.

**G6 — Numerical instability in joint training.**
Jointly optimizing restoration and detection is notoriously unstable.
Unbounded feature magnitudes from the restoration branch can destabilize
the detection head's box regression, and mixed-precision training can
introduce overflow-induced NaNs. Prior work does not systematically address
these practical obstacles, which are critical for reproducible, stable
training of multi-task adverse-weather detectors.

### A.2 Contributions of WRDNet

**C1 — Joint, feature-level fusion via the Feature Selection Gate (FSG).**
Rather than committing to a single restored image, WRDNet keeps the detection
backbone on the *original* foggy image and injects dehazed cues *at the
feature level* through a learned, per-pixel gating mechanism. This directly
addresses **G1** and **G2**: the network learns *where* dehazing is beneficial
rather than trusting it uniformly, and the gate is trained end-to-end with the
detection objective rather than a perceptual metric.

**C2 — Multi-density fog training.** We exploit the fact that Foggy Cityscapes
[12] provides each scene at three scattering coefficients
$\beta \in \{0.005, 0.01, 0.02\}$, tripling the effective training data and
enforcing fog-severity invariance. This addresses **G3** by compelling the
network to learn representations robust across the atmospheric continuum.

**C3 — Multi-level frequency-aware domain adaptation.** We bridge the
synthetic-to-real gap (**G4**) through three complementary mechanisms: Fourier
Domain Adaptation (FDA) at the input level, DCT-based feature alignment at the
representation level, and a novel FSG-consistency loss at the output level.
This multi-level strategy aligns the two domains without requiring real
annotations.

**C4 — Depth-guided feature fusion (DG-FSG).** We introduce the first
mechanism in which estimated monocular depth *actively modulates* how restored
and original features are combined, exploiting the physical coupling between
fog and depth (**G5**). The gate learns to trust dehazed features more for
distant objects and original features more for nearby objects, producing a
physically interpretable fusion policy.

**C5 — A numerically stable joint-training recipe.** We address **G6** through
a combination of bfloat16 mixed precision (which avoids the overflow-induced
NaNs of float16), a magnitude clamp on the fused features, and a two-phase
training schedule that first establishes a robust detection baseline before
enabling the fusion and domain-adaptation modules.

---

# II. Related Work

The design of WRDNet is grounded in a rich body of prior work on image
dehazing, adverse-weather detection, and domain adaptation. We organize this
review around the principal research directions that motivate our approach,
and we position WRDNet relative to each.

**Classical dehazing.** Early methods rely on hand-crafted priors, most
notably the Dark Channel Prior (DCP) [1] and its lightweight variants [2, 3,
4]. These methods estimate the transmission map and atmospheric light from
statistical priors and recover the scene radiance via the atmospheric
scattering model. While computationally efficient, they are prone to failure
under dense fog and overexposed regions, and they do not adapt to the
downstream detection task.

**Learning-based dehazing.** Deep learning methods replace hand-crafted priors
with learned mappings. Cycle-consistent adversarial networks [7] enable
unpaired dehazing, while transformer-based methods such as DehazeFormer [15]
and TCL-Net [17] achieve state-of-the-art restoration quality. However, these
methods are optimized for perceptual image-quality metrics rather than
downstream detection performance.

**Adverse-weather detection.** Recent work enhances detectors for foggy
conditions through attention and multi-scale fusion, including YOLOv8s-WAMNet
[16] and CAST-YOLO [14]. These methods improve robustness but do not jointly
optimize restoration and detection, and they do not exploit depth information.

**Domain adaptation.** To bridge the synthetic-to-real gap, Fourier Domain
Adaptation (FDA) [21] aligns input-level spectral statistics, while
feature-level alignment methods such as AdaDCP align representations in the
frequency domain. These methods address the domain gap but focus on a single
alignment level.

**Depth estimation.** Monocular depth estimation methods such as DPT and MiDaS
provide dense depth maps from a single image. Prior joint dehazing-detection
methods such as DEHRFormer and DCL estimate depth as a byproduct of dehazing
but do not use it to modulate feature fusion.

---

# III. Methodology

## A. Problem Formulation

We address the task of object detection under foggy driving conditions. Let
$\mathcal{I} \in \mathbb{R}^{H \times W \times 3}$ denote a foggy RGB image
captured by an onboard camera, and let $\mathcal{Y} = \{(\mathbf{b}_i, c_i)\}$
denote the set of ground-truth detections, where $\mathbf{b}_i \in
\mathbb{R}^4$ is the bounding box and $c_i \in \{0, \dots, C-1\}$ is the class
label over $C$ semantic categories. Our objective is to learn a mapping
$f_\theta: \mathcal{I} \mapsto \mathcal{Y}$ that remains accurate across a
continuum of atmospheric conditions.

Fog degrades the observed radiance through the atmospheric scattering model
[22]:

$$
\mathbf{I}(\mathbf{x}) = \mathbf{J}(\mathbf{x}) \, t(\mathbf{x}) + \mathbf{A} \,
\big(1 - t(\mathbf{x})\big),
$$

where $\mathbf{J}(\mathbf{x})$ is the clean scene radiance, $\mathbf{A}$ is the
global atmospheric light, and $t(\mathbf{x}) \in [0,1]$ is the transmission
map. Under a homogeneous atmosphere, the transmission decays exponentially
with scene depth $d(\mathbf{x})$:

$$
t(\mathbf{x}) = e^{-\beta \, d(\mathbf{x})},
$$

with $\beta > 0$ the scattering coefficient (fog density). This formulation
reveals a central tension for detection: aggressive dehazing can recover
texture in the far field but may amplify noise and hallucinate artifacts near
the camera, whereas conservative restoration preserves fidelity but leaves
distant objects obscured. A detector that commits to a single restored
representation therefore operates at a fixed point on this accuracy–fidelity
trade-off.

## A.1 What Makes WRDNet Weather-Resilient

The term *weather-resilient* refers to the ability of a detector to maintain
accurate performance across a continuum of atmospheric conditions, rather than
at a single operating point. WRDNet achieves this through four complementary
mechanisms that jointly address the spatial, severity, and domain dimensions of
weather variability.

**Spatial resilience via adaptive feature fusion.** Fog is spatially
non-uniform: near-field regions are only mildly degraded, while distant
regions are heavily obscured. A detector that commits to a single restored
image cannot adapt to this spatial variation. WRDNet's Feature Selection Gate
(FSG) learns a per-pixel weighting $\boldsymbol{\alpha}(\mathbf{x})$ that
selects between dehazed and original features at every location. In regions
where dehazing is reliable (distant, fog-obscured areas), the gate favors
restored features; where dehazing may introduce artifacts (near-field, thin
fog), it retains the original features. This spatial adaptivity is the first
pillar of weather resilience.

**Severity resilience via multi-density training.** Fog density $\beta$
varies continuously in real driving, from light mist to dense fog. A detector
trained at a single $\beta$ memorizes a specific atmospheric appearance and
fails to generalize. WRDNet trains on the multi-density structure of Foggy
Cityscapes, which renders each scene at $\beta \in \{0.005, 0.01, 0.02\}$.
This enforces fog-severity invariance: the network learns representations that
remain discriminative across the entire visibility continuum, rather than a
single operating point. This is the second pillar of weather resilience.

**Depth-conditional resilience via DG-FSG.** The atmospheric scattering model
couples fog severity to scene depth through $t(\mathbf{x}) =
e^{-\beta d(\mathbf{x})}$. WRDNet exploits this physical prior by conditioning
its fusion policy on estimated monocular depth. The Depth-Guided FSG (DG-FSG)
learns to trust dehazed features more for distant objects and original
features more for nearby objects, producing a fusion policy that is
physically consistent with the scattering model. This depth-conditional
behavior is the third pillar of weather resilience.

**Domain resilience via multi-level domain adaptation.** Synthetic fog differs
from real fog in its spectral and textural statistics. WRDNet bridges this
synthetic-to-real gap through a multi-level domain-adaptation strategy that
aligns the two domains at the input level (FDA), the feature level (DCT
alignment), and the output level (FSG consistency). This ensures that the
learned representations transfer to real-world fog, which is the fourth pillar
of weather resilience.

Together, these four mechanisms enable WRDNet to maintain detection accuracy
across varying fog density, spatial extent, and domain shift, which is the
defining property of a weather-resilient detector.

## B. Architectural Overview

We propose the **Weather-Resilient Detection Unified Network (WRDNet)**, a
multi-branch architecture that jointly performs restoration, detection, and
depth estimation while adaptively fusing features across the restoration and
detection streams. The complete model is illustrated in Fig. 1 and comprises
four principal components:

1. **Restoration branch** — a lightweight DehazeFormer-T [15]
   transformer that estimates the clean image $\hat{\mathbf{J}}$ and exposes
   multi-scale encoder features.
2. **Detection branch** — a YOLOv11s [25] detector whose
   backbone operates on the *original* foggy image to preserve spatial
   resolution and small-object cues.
3. **Feature Selection Gate (FSG)** — a learned, per-pixel gating mechanism
   that fuses restoration and detection features at three scales.
4. **Depth decoder** — a lightweight progressive upsampling head that
   estimates monocular depth to guide the fusion (DG-FSG variant).

A key design decision is that the detection backbone consumes the **original
foggy image** rather than the dehazed output. This preserves the native
resolution and high-frequency content required for detecting small and distant
objects, which are precisely the most vulnerable under fog. The restored
representation is injected *at the feature level* through the FSG, allowing the
network to selectively exploit dehazed cues where they are beneficial without
committing to a single restored image.

## C. Feature Selection Gate (FSG)

The central contribution of WRDNet is the Feature Selection Gate, which learns
a spatially varying interpolation between restoration and detection features.
Let $\mathbf{F}^r \in \mathbb{R}^{C \times H_s \times W_s}$ and $\mathbf{F}^o
\in \mathbb{R}^{C \times H_s \times W_s}$ denote the restoration and original
feature maps at scale $s$, respectively. The FSG produces a gating map
$\boldsymbol{\alpha} \in [0,1]^{1 \times H_s \times W_s}$ via a small
convolutional gating network $g_\phi$:

$$
\boldsymbol{\alpha} = \sigma\Big( g_\phi\big( [\mathbf{F}^r; \, \mathbf{F}^o]
\big) \Big),
$$

where $[\cdot;\cdot]$ denotes channel-wise concatenation and $\sigma$ is the
sigmoid activation. The fused feature is then computed as a convex combination:

$$
\mathbf{F}^f = \boldsymbol{\alpha} \odot \mathbf{F}^r + (1 -
\boldsymbol{\alpha}) \odot \mathbf{F}^o,
$$

where $\odot$ is the Hadamard product. Because $\boldsymbol{\alpha} \in
[0,1]$, the fused feature lies on the line segment between the two branches,
guaranteeing that the gate can never amplify or invert the input statistics.
Intuitively, $\alpha \to 1$ selects the dehazed representation (beneficial for
distant, fog-obscured regions), while $\alpha \to 0$ retains the original
features (beneficial for near-field regions where dehazing may introduce
artifacts).

To enrich the gating context, we integrate a **Cross-Dimensional Multi-Scale
Attention (CDMSA)** module that computes channel, spatial, and cross-scale
attention over the concatenated features. The cross-scale term propagates
information from coarser scales to refine the gating decision at finer scales.

**Stability analysis.** A critical practical concern is the numerical
stability of the fused features when propagated through the detection head.
The YOLO detection head employs Distribution Focal Loss (DFL) [23], which
predicts a discrete distribution over box-edge offsets. If the
fused features exhibit unbounded magnitude, the DFL logits can saturate,
causing the predicted box edges to collapse to a degenerate configuration. To
prevent this, we apply a magnitude clamp after normalization:

$$
\tilde{\mathbf{F}}^f = \operatorname{clamp}\big( \operatorname{BN}(
\mathbf{F}^f ), -\tau, \tau \big),
$$

with $\tau = 10$. This bounds the feature magnitude to a range compatible with
the pretrained detection head while preserving the relative ordering of the
gated fusion. We also initialize the gating network's final bias to $-2.0$,
which yields $\sigma(-2.0) \approx 0.12$, biasing the gate toward the original
features early in training and preventing the detection loss from spiking on
unstable restored features.

## D. Depth-Guided FSG (DG-FSG)

For the depth-aware variant, we augment the gating network with a monocular
depth estimate. A lightweight depth decoder $\mathcal{D}$ progressively
upsamples a bottleneck feature to produce a dense depth map
$\hat{d} \in \mathbb{R}^{H \times W}$. The depth encoding is resized to each
scale and concatenated with the restoration and original features:

$$
\boldsymbol{\alpha} = \sigma\Big( g_\phi\big( [\mathbf{F}^r; \, \mathbf{F}^o;
\, \mathcal{E}(\hat{d})] \big) \Big),
$$

where $\mathcal{E}$ is a depth encoder. This formulation exploits the physical
prior that fog severity increases with depth: distant regions (large $d$)
benefit more from dehazing, so the gate can learn a depth-conditional fusion
policy. The depth decoder is supervised with a scale-invariant loss [24] on
synthetic data.

## E. Multi-Density Fog Training

A central limitation of prior foggy-driving benchmarks is that they train on a
single fog density $\beta$, causing the detector to memorize a specific
atmospheric appearance. We instead exploit the fact that the Foggy Cityscapes
dataset [12] provides the same scene rendered at multiple
scattering coefficients $\beta \in \{0.005, 0.01, 0.02\}$. Training on all
three densities simultaneously:

1. **Triples the effective training data**, mitigating overfitting on the
   limited set of annotated scenes;
2. **Enforces fog-severity invariance**, compelling the network to learn
   representations that are robust across the atmospheric continuum rather
   than a single operating point.

Formally, the training set is
$\mathcal{D} = \bigcup_{\beta \in \mathcal{B}} \mathcal{D}_\beta$, where
$\mathcal{B} = \{0.005, 0.01, 0.02\}$ and $\mathcal{D}_\beta$ is the set of
images rendered at density $\beta$. This multi-density strategy is a
principled, physically grounded form of data expansion that directly targets
the task's core challenge.

## E.1 Datasets

WRDNet is trained and evaluated on a combination of synthetic and real-world
adverse-weather datasets. Table I summarizes the datasets, their roles, and
their citations.

**Table I: Datasets used in WRDNet.**

| Dataset | Type | Split | Role | Citation |
|---------|------|-------|------|----------|
| Foggy Cityscapes | Synthetic fog (labeled) | train + val | Supervised detection + restoration + depth | [12] |
| ACDC | Real fog (unlabeled) | train | Domain adaptation (FDA, DCT, FSG-consistency) | [26] |
| ACDC | Real fog (labeled) | val | Validation (mAP monitoring) | [26] |
| Foggy Driving | Real fog (labeled) | test | Final evaluation (test mAP) | [12] |
| Foggy Zurich | Real fog (unlabeled) | test | Cross-domain generalization | [27] |

**Synthetic training data.** The primary supervised training signal comes from
the Foggy Cityscapes dataset [12], which renders each Cityscapes scene at three
scattering coefficients $\beta \in \{0.005, 0.01, 0.02\}$. We use both the
`train` and `val` splits for training (3,475 base scenes × 3 densities = 10,425
images), since our final evaluation is on the disjoint ACDC and Foggy Driving
datasets. Each synthetic sample provides a foggy image, its clear ground truth,
a disparity-derived depth map, and instance-level bounding-box labels for eight
driving-relevant classes.

**Real-world domain-adaptation data.** To bridge the synthetic-to-real gap, we
use the ACDC dataset [26] as the unlabeled target domain during Phase 1. The
ACDC `train` split provides real foggy images without labels, which drive the
three domain-adaptation losses (FDA, DCT alignment, and FSG consistency). The
ACDC `val` split, which has labels, is used to monitor mAP during training.

**Real-world evaluation data.** For the final evaluation, we use the Foggy
Driving dataset [12], a standard benchmark of real foggy driving scenes with
bounding-box annotations, and the Foggy Zurich dataset [27] to assess
cross-domain generalization to an unseen real-world fog distribution.

## F. Domain Adaptation

To bridge the gap between synthetic fog and real-world fog, we incorporate
unsupervised domain adaptation using real foggy images from the ACDC dataset
[26]. The synthetic branch provides supervised detection
supervision, while the real branch provides an unlabeled domain signal through
three complementary losses:

1. **Frequency Domain Adaptation (FDA)** [21] — swaps the
   low-frequency components of synthetic and real images to align their
   spectral statistics.
2. **Discrete Cosine Transform (DCT) alignment** — aligns the feature
   distributions of the two domains in the frequency domain via a
   maximum-mean-discrepancy (MMD) loss.
3. **FSG consistency** — encourages the gating maps produced on synthetic and
   real images to be consistent given their estimated fog density, promoting
   domain-invariant fusion.

The total training objective is a weighted sum of the detection loss
$\mathcal{L}_{\text{det}}$, restoration loss $\mathcal{L}_{\text{rest}}$,
depth loss $\mathcal{L}_{\text{depth}}$, and domain losses:

$$
\mathcal{L} = \mathcal{L}_{\text{det}} + \lambda_r \mathcal{L}_{\text{rest}} +
\lambda_d \mathcal{L}_{\text{depth}} + \lambda_{\text{dom}}
\mathcal{L}_{\text{dom}} + \lambda_{\text{fsg}} \mathcal{L}_{\text{fsg}},
$$

where the domain weight $\lambda_{\text{dom}}$ is linearly ramped from $0$ to
its target value over the first epochs to avoid destabilizing the detector
early in training.

## G. Training Procedure

WRDNet is trained in two stages. In **Phase 0**, the restoration branch is
frozen and the detection branch is trained on the multi-density synthetic data
with the FSG bypassed, establishing a robust detection baseline. In **Phase 1**,
all components are unfrozen, the FSG is enabled with the magnitude clamp, and
domain adaptation is applied using real foggy images. We use the AdamW
optimizer with a cosine learning-rate schedule and gradient clipping at unit
norm. Mixed-precision training is performed in bfloat16, which provides the
exponent range of float32 (preventing overflow in the gating convolutions)
while retaining the throughput benefits of reduced precision on modern
tensor-core hardware.

---

## References

[1] K. He, J. Sun, and X. Tang, "Single image haze removal using dark channel
prior," *IEEE Transactions on Pattern Analysis and Machine Intelligence
(TPAMI)*, vol. 33, no. 12, pp. 2341–2353, 2011.

[2] C.-C. Sun, N.-H.-H. Pham, A. A. Bryantono, and J.-W. Hsieh, "Lightweight
computation single-image fog removal based on a new improved adaptive dark
channel prior," *IEEE Transactions on Intelligent Transportation Systems
(T-ITS)*, vol. 26, no. 11, 2025.

[3] "A fast method of fog and haze removal," *IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP)*, 2016.

[4] X. Sang, Y. Yang, and X. Hou, "Fog removal method of slope monitoring
image based on vision detection," in *Proceedings of the 37th Chinese Control
Conference (CCC)*, 2018, pp. 1–6.

[5] I. Ogunrinde, "A review of the impacts of defogging on deep learning-based
object detectors in self-driving cars," in *IEEE SoutheastCon*, 2021.

[6] N. U. A. Tahir, Z. Zhang, M. Asim, J. Chen, and M. ELAffendi, "Object
detection in autonomous vehicles under adverse weather: A review of
traditional and deep learning approaches," *Algorithms*, vol. 17, no. 3, 2024.

[7] W. Liu, X. Hou, J. Duan, and G. Qiu, "End-to-end single image fog removal
using enhanced cycle consistent adversarial networks," *IEEE Transactions on
Image Processing (TIP)*, vol. 29, pp. 7819–7833, 2020.

[8] H. Lim, J. Lee, H. Kim, H. Oh, and J. Paik, "Image enhancement for
high-resolution visual contents," in *IEEE International Conference on
Electronics, Information, and Communication (ICEIC)*, 2023.

[9] S. H. Gangolli and A. J. L. Fonseca, "Image enhancement using various
histogram equalization techniques," in *IEEE Global Conference for Advancement
in Technology (GCAT)*, 2019.

[10] Erwin, A. Nevriyanto, and D. Purnamasari, "Image enhancement using the
image sharpening, contrast enhancement, and standard median filter (noise
removal) with pixel-based and human visual system-based measurements," in
*IEEE International Conference on Electrical Engineering and Computer Science
(ICECOS)*, 2017.

[11] C.-Y. Lee, "Design a hardware applying fog removal algorithm using median
dark channel prior for autonomous driving car," in *International Conference on
Computational Science and Computational Intelligence (CSCI)*, 2023.

[12] C. Sakaridis, D. Dai, and L. Van Gool, "Semantic foggy scene
understanding with synthetic data," *International Journal of Computer Vision
(IJCV)*, vol. 126, no. 9, pp. 973–991, 2018. [Online]. Available:
https://arxiv.org/abs/1708.07819

[13] S. Gharatappeh, S. Y. Sekeh, and V. Dhiman, "Weather-aware object
detection transformer for domain adaptation (FogAwareAttention)," arXiv
preprint arXiv:2504.10877, 2025.

[14] X. Liu, B. Zhang, and N. Liu, "CAST-YOLO: An improved YOLO based on a
cross-attention strategy transformer for foggy weather adaptive detection,"
*Applied Sciences*, 2024.

[15] Y. Song, Z. He, H. Qian, and X. Du, "Vision transformers for single image
dehazing," *IEEE Transactions on Image Processing (TIP)*, vol. 32, pp.
1927–1941, 2023. [Online]. Available: https://arxiv.org/abs/2204.03883

[16] M. Jaiswal, K. K. Nagwanshi, M. Heenaye-Mamode Khan, U. Verma, and A.
Taylor, "YOLOv8s-WAMNet: Enhancing robust vehicle detection under adverse
weather via hybrid attention and multi-scale fusion in real time," *Scientific
Reports*, 2026.

[17] C. Tang and W. Lou, "TCL-Net: A lightweight and efficient dehazing
network with frequency-domain fusion and multi-angle attention," in *ACCV*,
2024.

[18] Y. Xie, H. Wei, Z. Liu, X. Wang, and X. Ji, "SynFog: A photo-realistic
synthetic fog dataset based on end-to-end imaging simulation for advancing
real-world defogging in autonomous driving," arXiv preprint arXiv:2403.17094,
2024.

[19] A. Aryashad, P. Razmara, A. Mahjoub, S. Azizi, M. Salmani, and A.
Firouzkouhi, "From filters to VLMs: Benchmarking defogging methods through
object detection and segmentation performance," arXiv preprint
arXiv:2510.03906, 2026.

[20] N. Raza, M. A. Habib, M. Ahmad, Q. Abbas, M. B. Aldajani, and M. A. Latif,
"Efficient and cost-effective vehicle detection in foggy weather for
edge/fog-enabled traffic surveillance and collision avoidance systems,"
*Computers, Materials & Continua*, 2024.

[21] Y. Yang and S. Soatto, "FDA: Fourier domain adaptation for semantic
segmentation," in *Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR)*, 2020. [Online]. Available:
https://arxiv.org/abs/2004.05498

[22] H. Koschmieder, "Theorie der horizontalen Sichtweite," *Beiträge zur
Physik der freien Atmosphäre*, vol. 12, pp. 33–53, 1924.

[23] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang,
"Generalized focal loss: Learning qualified and distributed bounding boxes for
dense object detection," in *Advances in Neural Information Processing Systems
(NeurIPS)*, 2020.

[24] D. Eigen, C. Puhrsch, and R. Fergus, "Depth map prediction from a single
image using a multi-scale deep network," in *Advances in Neural Information
Processing Systems (NeurIPS)*, 2014.

[25] G. Jocher and J. Qiu, "Ultralytics YOLOv11," 2024. [Online]. Available:
https://github.com/ultralytics/ultralytics

[26] C. Sakaridis, D. Dai, and L. Van Gool, "ACDC: The adverse conditions
dataset with correspondences for semantic driving scene understanding," in
*Proceedings of the IEEE/CVF International Conference on Computer Vision
(ICCV)*, 2021.

[27] C. Sakaridis, D. Dai, S. Hecker, and L. Van Gool, "Model adaptation with
synthetic and real data for semantic dense foggy scene understanding," in
*European Conference on Computer Vision (ECCV)*, 2018, pp. 707–724.
