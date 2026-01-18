# Rotation-Invariant Texture Classification using Local Binary Patterns

A Python implementation of rotation-invariant texture classification using Local Binary Patterns (LBP) and its variants, achieving **100% accuracy** on the Brodatz rotated texture dataset.

![Bark Texture Rotations](bark_rotations.png)
*Example: Bark texture at different rotation angles from the Brodatz dataset*


##  Overview

This project implements a robust texture classification system that is invariant to rotation transformations. The system uses Local Binary Patterns (LBP), a powerful texture descriptor, combined with multi-resolution analysis and local variance features.

**Key Achievement:** 100% classification accuracy on 13 Brodatz textures across 7 rotation angles (0°, 30°, 60°, 90°, 120°, 150°, 200°).

📄 **For detailed methodology and complete analysis, please refer to:**
- [**Implementation Code**](22b2505_finalProject_CS663.py) - Complete source code
- [**Project Presentation**](22b2505_finalProject_CS663.pdf) - Detailed slides with methodology and references

##  Features

- **Multiple LBP Variants:**
  - Basic LBP with circular neighborhoods
  - Rotation-invariant LBP (LBPri)
  - Rotation-invariant uniform LBP (LBPriu2)
  
- **Multi-Resolution Analysis:**
  - Multiple (P, R) configurations: (8,1), (16,2), (24,3)
  - Combined multi-scale operators
  
- **Variance Descriptors:**
  - Local variance (VAR) computation
  - Joint LBP/VAR histograms for enhanced discrimination
  
- **Efficient Implementation:**
  - Vectorized NumPy operations
  - Pre-computed lookup tables for rotation invariance
  - Custom bilinear interpolation for circular neighborhoods

##  Results

### Classification Accuracy

| Operator | Train 0° | Train 30° | Train 60° | Train 90° | **Average** |
|----------|----------|-----------|-----------|-----------|-------------|
| LBP₈,₁ | 88.46% | 89.74% | 84.62% | 83.33% | **86.54%** |
| LBP₁₆,₂ | 93.59% | 96.15% | 94.87% | 94.87% | **94.87%** |
| LBP₂₄,₃ | 100.00% | 98.72% | 100.00% | 100.00% | **99.68%** |
| **LBP₁₆,₂/VAR** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| LBP₂₄,₃/VAR | 98.72% | 100.00% | 100.00% | 100.00% | **99.68%** |
| LBP₈,₁+₁₆,₂ | 92.31% | 93.59% | 87.18% | 92.31% | **91.35%** |

### Key Findings

1. **Multi-resolution improves performance:** Larger neighborhoods (P=16, R=2 or P=24, R=3) capture coarser spatial structures
2. **VAR is highly effective:** Adding local variance complements LBP by encoding contrast information
3. **Perfect classification achieved:** LBP₁₆,₂/VAR achieves 100% accuracy across all training scenarios



## 📁 Dataset

### Brodatz Rotated Textures

The project uses 13 textures from the [USC-SIPI Brodatz rotated texture database](http://sipi.usc.edu/database/database.php?volume=textures):

- **Textures:** bark, brick, bubbles, grass, leather, pigskin, raffia, sand, straw, water, weave, wood, wool
- **Rotations:** 0°, 30°, 60°, 90°, 120°, 150°, 200° (7 angles per texture)
- **Format:** Grayscale, 512×512 pixels, 8-bit TIFF
- **Total images:** 91 (13 textures × 7 angles)

### Data Organization

```
rotate/
├── bark.000.tiff
├── bark.030.tiff
├── bark.060.tiff
...
├── wool.200.tiff
```

### Training Protocol

- **Training:** Use one rotation angle per texture (e.g., 0°)
- **Testing:** Evaluate on remaining 6 rotation angles
- **Patch extraction:** 16×16 subimages from 512×512 images
- **Cross-validation:** Train on 0°, 30°, 60°, 90° separately

## 📂 Project Structure

```
rotation-invariant-texture-classification/
│
├── 22b2505_finalProject_CS663.py  # Main implementation
├── README.md                       # This file
│
├── docs/
│   ├── 22b2505_finalProject_CS663.pdf  # Detailed presentation
│   └── images/                     # Documentation images
│
├── rotate/                         # Dataset directory (not included)
│   ├── bark.000.tiff
│   └── ...
│
└── 22b2505_results/               # Generated results (after running)
    ├── lbp_maps/
    ├── histograms/
    ├── confusion_matrices/
    └── experiment_summary.txt
```

## 🔧 Implementation Details

### Key Optimizations

1. **Lookup Tables (LUTs):**
   - Pre-computed rotation-invariant mappings
   - RIU2 pattern mappings
   - Avoids expensive per-pixel recomputation

2. **Vectorized Operations:**
   - NumPy broadcasting for neighbor comparisons
   - Batch processing of all pixels simultaneously

3. **Bilinear Interpolation:**
   - Custom implementation for circular neighborhoods
   - Handles non-integer pixel coordinates efficiently

4. **Variance Quantization:**
   - Percentile-based cut-points from training data
   - Consistent binning across train/test sets

5. **Numerical Stability:**
   - Laplace smoothing for histogram models
   - Log-probabilities to prevent underflow

### Class Structure

- **`LBPOperator`**: Core LBP computation with configurable P, R
- **`TextureClassifier`**: Multi-operator training and classification
- **Helper functions**: Image loading, patch extraction, visualization



## 📚 References

1. T. Ojala, M. Pietikäinen and T. Mäenpää, "Multiresolution Gray-scale and Rotation Invariant Texture Classification with Local Binary Patterns," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 7, pp. 971-987, 2002. [IEEE Xplore](https://ieeexplore.ieee.org/document/1017623)

2. Brodatz Texture Database, USC-SIPI Image Database. [Link](http://sipi.usc.edu/database/database.php?volume=textures)
