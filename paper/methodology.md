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
[Koschmieder, 1924]:

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

## B. Architectural Overview

We propose the **Weather-Resilient Detection Unified Network (WRDNet)**, a
multi-branch architecture that jointly performs restoration, detection, and
depth estimation while adaptively fusing features across the restoration and
detection streams. The complete model is illustrated in Fig. 1 and comprises
four principal components:

1. **Restoration branch** — a lightweight DehazeFormer-T [Song et al., 2023]
   transformer that estimates the clean image $\hat{\mathbf{J}}$ and exposes
   multi-scale encoder features.
2. **Detection branch** — a YOLOv11s [Ultralytics, 2024] detector whose
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
The YOLO detection head employs Distribution Focal Loss (DFL) [Li et al.,
2022], which predicts a discrete distribution over box-edge offsets. If the
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
policy. The depth decoder is supervised with a scale-invariant loss
[Eigen et al., 2014] on synthetic data.

## E. Multi-Density Fog Training

A central limitation of prior foggy-driving benchmarks is that they train on a
single fog density $\beta$, causing the detector to memorize a specific
atmospheric appearance. We instead exploit the fact that the Foggy Cityscapes
dataset [Sakaridis et al., 2018] provides the same scene rendered at multiple
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

## F. Domain Adaptation

To bridge the gap between synthetic fog and real-world fog, we incorporate
unsupervised domain adaptation using real foggy images from the ACDC dataset
[Sakaridis et al., 2021]. The synthetic branch provides supervised detection
supervision, while the real branch provides an unlabeled domain signal through
three complementary losses:

1. **Frequency Domain Adaptation (FDA)** [Yang and Soatto, 2020] — swaps the
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
