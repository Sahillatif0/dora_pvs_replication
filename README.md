# Replication of DoRA-PVS Winning Methodology (VICOROBIGR)

This project replicates the winning methodology (**VICOROBIGR**) from the **DoRA-PVS Challenge (AIiH 2026)**:
*Assessing Generalisation of Perivascular Space Segmentation Across Heterogeneous MRI Cohorts*.

---

## Directory Structure

```text
dora_pvs_replication/
├── configs/              # Hyperparameters, randomisation ranges, training configs
├── data/
│   ├── atlases/          # Anatomical segmentations (SynthSeg / NextBrain / FastSurfer)
│   └── synthetic/        # Generated synthetic volumes & multi-class masks
├── src/
│   ├── generator/        # DRIPS-style procedural PVS + Pathology + MRI physics simulation
│   ├── models/           # DynUNet / nnU-Net architecture and multi-class loss functions
│   └── evaluation/       # AUPRC, clDice (centreline Dice), lesion-wise DSC metrics
├── requirements.txt      # Python dependencies
└── README.md
```

---

## The 4-Step Replication Plan

### Step 1: Anatomical Foundation & Label Preparation
* Acquire or prepare anatomical label maps (White Matter, Gray Matter, Basal Ganglia, CSF).
* Can be derived using FastSurfer or SynthSeg on public datasets (e.g., ADNI, IXI, OASIS).

### Step 2: Procedural Generator (DRIPS + Pathology Simulation)
* **PVS Simulation**: Seed 3D curved tubular structures in Centrum Semiovale (CSO) and Basal Ganglia (BG) with smooth cosine cross-sections.
* **Pathology Simulation (Key to VICOROBIGR)**: Seed white matter hyperintensity (WMH) confounder blobs to distinguish PVS from lesions.
* **Domain Randomisation**: Randomize contrast (T1-like, T2-like, intermediate), apply affine spatial deformations, Gaussian blurring, Rician/Gaussian noise, and simulate thick anisotropic 2D slice acquisition (0.5 mm to 2.0 mm).

### Step 3: Multi-Class Segmentation Model (DynUNet)
* 3-Class output: `[0: Background, 1: PVS, 2: Pathology]`
* Loss function: Combined Soft Dice + Focal / Cross-Entropy loss.
* Multi-scale training (native simulated resolution + 0.5 mm isotropic resolution).

### Step 4: Out-of-Sample Evaluation Pipeline
* Calculate:
  1. **AUPRC** (Area Under Precision-Recall Curve)
  2. **clDice** (Topology/Centreline Dice)
  3. **Lesion-wise DSC** (Instance-level detection via one-inside criterion)
