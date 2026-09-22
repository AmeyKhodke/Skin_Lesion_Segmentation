# Skin Lesion Segmentation — Computer-Aided Diagnosis

A deep learning pipeline for automated skin lesion boundary segmentation from dermoscopic images, built on the ISIC 2018 Challenge (Task 1) dataset. This project targets the first-stage problem in computer-aided diagnosis (CAD) systems: precisely localizing a lesion before any downstream diagnostic feature extraction (asymmetry, border irregularity, color variation) can occur.

## Motivation

Automated lesion segmentation supports:
- **Early melanoma screening** — segmentation enables automatic extraction of the ABCDE diagnostic features dermatologists use manually
- **Teledermatology** — remote/preliminary assessment where specialist access is limited
- **Clinical trial outcome measurement** — objective, quantified lesion/wound area tracking over time

Precise boundaries matter clinically, not just technically: irregular, jagged lesion borders are themselves a diagnostic indicator of malignancy, not merely a preprocessing artifact.

## Dataset

**ISIC 2018 Challenge — Task 1 (Lesion Boundary Segmentation)**
- ~2,594 dermoscopic training images (JPEG) with corresponding binary ground truth masks (PNG)
- 100 validation images, 1,000 test images
- Source: [ISIC Archive](https://challenge.isic-archive.com/data/#2018)

## Architecture

**U-Net** (Ronneberger et al., 2015) — an encoder-decoder convolutional network with skip connections, originally designed for biomedical image segmentation.

- **Contracting path:** 4× [Conv2D → ReLU → Conv2D → ReLU → MaxPooling], filters doubling (64 → 128 → 256 → 512), progressively reducing spatial resolution while extracting context
- **Bottleneck:** Conv2D → ReLU → Conv2D → ReLU (1024 filters)
- **Expansive path:** 4× [Conv2DTranspose (up-convolution) → Concatenate with corresponding contracting-path feature map (skip connection) → Conv2D → ReLU → Conv2D → ReLU], restoring spatial resolution while recovering fine boundary detail via skip connections
- **Output:** 1×1 Conv2D with sigmoid activation → per-pixel lesion probability map

**Training setup:**
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Input size: 256×256×3
- Callbacks: `ModelCheckpoint` (best val_loss), `EarlyStopping`, `ReduceLROnPlateau`
- Stack: Python, TensorFlow/Keras, OpenCV

## Results

| Metric | Score |
|---|---|
| Mean Dice Coefficient | **0.8611** |
| Mean IoU | **0.7824** |
| Std (Dice) | 0.1604 |

![Prediction Samples](images/prediction_samples.png)

Predictions closely track ground truth boundaries across a range of lesion sizes, colors, and image conditions (including hair occlusion and low contrast).

## Error Analysis

While overall performance is strong, a subset of validation cases score near-zero Dice. Inspecting these cases reveals a consistent pattern: **the model fails specifically on small lesions** — well-defined, correctly-labeled lesions that occupy a small fraction of the image area.

![Worst Cases](images/worst_cases.png)

**Hypothesis:** the network's contracting path downsamples spatial resolution by a factor of 16 (256 → 16 at the bottleneck) before the deepest representation is formed. For small lesions, this aggressive downsampling can compress the lesion's spatial signal to only a few pixels or lose it entirely before it reaches the bottleneck — larger lesions retain enough signal to survive this compression, small ones do not. This is a documented limitation of standard U-Net on small-object segmentation.

## Additional Experiments (Not Adopted)

To address the small-lesion failure mode, two targeted interventions were tested:

1. **Attention U-Net** — added attention gates on each skip connection to suppress irrelevant background activation and emphasize salient regions before concatenation.
2. **Tversky Loss** (α=0.3, β=0.7) — reweighted the loss to penalize false negatives more heavily than false positives, directly targeting the "missed lesion entirely" failure pattern.

Neither intervention, alone or combined, outperformed the baseline result above (best combined result: 0.8497 Dice). This suggests the small-lesion failure mode may require a different approach — such as multi-scale input processing, targeted oversampling of small-lesion training examples, or longer training convergence — rather than architecture or loss reweighting alone. Noted as future work.

## Project Structure

```
skin-lesion-segmentation/
├── notebook.ipynb
├── README.md
├── best_unet_model.keras
└── images/
    ├── prediction_samples.png
    ├── worst_cases.png
    └── loss_curve.png
```

## Future Work

- Multi-scale / patch-based training to preserve small-lesion signal
- Targeted oversampling of small-lesion training examples
- Extend to lesion classification (benign/malignant) using segmented ROI as input to a second-stage classifier
- Deploy as a Streamlit demo for interactive screening-support visualization

## Disclaimer

This is a research/academic prototype, not a validated medical device. Outputs are not intended for clinical diagnosis.
