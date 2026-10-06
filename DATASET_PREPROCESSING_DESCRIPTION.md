# Dataset Analysis & Preprocessing Configuration Guide

## Part 1: ICCAD-12 Dataset Overview

### 1.1 Dataset Source & Acquisition

**Official Source:** International Conference on Computer-Aided Design (ICCAD)  
**Benchmark:** ICCAD-12 (2012)  
**Download Link:** http://www.ispd.cc/contests/12/%5D(https://drive.google.com/file/d/1jx7gDR92sqoIw2Nh4NGwwZwC-qNp2osd/view)  
**Format:** RAR archive (iccad_official.rar)

### 1.2 Dataset Statistics

#### Complete Breakdown

| Benchmark | Train HS | Train NHS | Train Total | Test HS | Test NHS | Test Total | Total | HS % (Train) | HS % (Test) | Imbalance Ratio |
|-----------|----------|-----------|------------|---------|----------|-----------|-------|-------------|------------|-----------------|
| ICCAD-1   | 99       | 340       | 439        | 226     | 4,679    | 4,905     | 5,344 | 22.6%       | 4.6%       | 3.4:1           |
| ICCAD-2   | 174      | 5,285     | 5,459      | 498     | 41,298   | 41,796    | 47,255 | 3.2%       | 1.2%       | 30.4:1          |
| ICCAD-3   | 909      | 4,643     | 5,552      | 1,808   | 46,333   | 48,141    | 53,693 | 16.4%       | 3.8%       | 5.1:1           |
| ICCAD-4   | 95       | 4,452     | 4,547      | 177     | 31,890   | 32,067    | 36,614 | 2.1%        | 0.6%       | 46.9:1          |
| ICCAD-5   | 26       | 2,716     | 2,742      | 41      | 19,327   | 19,368    | 22,110 | 0.9%        | 0.2%       | 104.5:1         |
| **TOTAL** | **1,303**| **17,436**| **18,739** | **2,750** | **143,527** | **146,277** | **165,016** | **6.9%** | **1.9%** | **13.0:1 avg** |

#### Key Observations:

1. **Severe Class Imbalance:**
   - Average: 93.1% non-hotspots, 6.9% hotspots
   - Range: 0.9% (ICCAD-5) to 22.6% (ICCAD-1)
   - Naive accuracy baseline (predict all NHS): 93-99% (misleading!)

2. **Benchmark Difficulty Variation:**
   - **Easiest:** ICCAD-1 (most balanced, 22.6% HS)
   - **Hardest:** ICCAD-5 (most imbalanced, 0.9% HS, only 26 training hotspots)

3. **Data Sufficiency:**
   - ICCAD-1,2,3: Adequate training data (99-909 hotspots)
   - ICCAD-4,5: Insufficient training data (26-95 hotspots)

### 1.3 Image Specifications

```
Image Format:     PNG (Portable Network Graphics)
Compression:      Lossless
Color Space:      Grayscale (Mode L)
Bit Depth:        8-bit (0-255 intensity values)
Dimensions:       1200 × 1200 pixels
Pixels/Image:     1,440,000
File Size:        ~120-150 KB per image
Encoding:         Standard PNG with gamma correction

Directory Structure:
benchmark/
├── train/
│   ├── train_hs/          # Training hotspots
│   │   ├── HS0.png
│   │   ├── HS1.png
│   │   └── ... (HS{N}.png)
│   └── train_nhs/         # Training non-hotspots
│       ├── NHS0.png
│       ├── NHS1.png
│       └── ... (NHS{N}.png)
└── test/
    ├── test_hs/           # Test hotspots
    │   ├── HS0.png
    │   └── ... (HS{N}.png)
    └── test_nhs/          # Test non-hotspots
        ├── NHS0.png
        └── ... (NHS{N}.png)
```

### 1.4 Physical Interpretation

**Hotspots (HS):** 
- Regions where printed layout deviates from design intent
- Two types:
  - **Pinching hotspots:** Insufficient metal width → open circuits
  - **Bridging hotspots:** Excessive metal width → short circuits
- Caused by diffraction effects at sub-wavelength feature sizes

**Non-Hotspots (NHS):**
- Regions with normal manufacturing behavior
- Reliable electrical characteristics
- Good yield prediction

---

## Part 2: Data Preprocessing Pipeline

### 2.1 Image Loading

```python
from PIL import Image
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    """
    Load image in grayscale mode.
    
    Args:
        image_path: Path to PNG image
    
    Returns:
        Grayscale image as numpy array (dtype: uint8, range 0-255)
    """
    image = Image.open(image_path).convert('L')  # 'L' = grayscale
    return np.array(image, dtype=np.uint8)
```

**Key Points:**
- Always convert to grayscale ('L' mode), even if PNG is color
- Preserve original 0-255 range for edge detection
- Load as uint8 initially (more efficient than float32)

### 2.2 Normalization

```python
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range for neural network input.
    
    Args:
        image: uint8 grayscale image (0-255)
    
    Returns:
        Normalized float32 image (0-1)
    """
    image_float = image.astype(np.float32)
    image_normalized = image_float / 255.0
    return image_normalized

# Usage
image_uint8 = load_image("HS0.png")
image_normalized = normalize_image(image_uint8)
print(f"Original range: {image_uint8.min()}-{image_uint8.max()}")
print(f"Normalized range: {image_normalized.min()}-{image_normalized.max()}")
# Output: Original range: 0-255, Normalized range: 0.0-1.0
```

**Rationale:**
- Neural networks train better on [0, 1] range
- Normalizes gradient magnitudes
- Improves numerical stability

### 2.3 Three Representations

#### Representation 1: Original

```python
def get_original_representation(image: np.ndarray) -> np.ndarray:
    """
    Original representation: raw normalized layout.
    
    What it captures:
    - Actual feature geometry (shapes, sizes)
    - Feature spacing and proximity
    - Layout topology and connectivity
    
    Processing:
    - Simple normalization (no transformation)
    
    Shape: (H, W) or (C, H, W) for PyTorch
    """
    return image  # Already normalized
```

**Characteristics:**
- Captures complete layout information
- Contains all relevant details
- May include noise and irrelevant patterns
- Reference approach; forms baseline

#### Representation 2: Edge Detection

```python
import cv2

def get_edge_representation(image: np.ndarray) -> np.ndarray:
    """
    Edge representation: Canny edge detection.
    
    What it captures:
    - Layout boundaries and transitions
    - Geometric edges where lithographic stress concentrates
    - Sharp corners and feature discontinuities
    
    Processing:
    1. Convert to uint8 if necessary
    2. Apply Canny edge detection
    3. Normalize to [0, 1]
    
    Canny Parameters:
    - Low threshold: 50 (edges below this are discarded)
    - High threshold: 150 (edges above this are accepted)
    - Kernel size: 3 (implicit in cv2.Canny)
    
    Shape: (H, W) binary edge map
    """
    # Ensure uint8 format for Canny
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Apply Canny edge detection
    edges = cv2.Canny(image_uint8, threshold1=50, threshold2=150)
    
    # Normalize to [0, 1]
    edges_normalized = edges.astype(np.float32) / 255.0
    
    return edges_normalized

# Usage
image = load_image("HS0.png")
image_norm = normalize_image(image)
edges = get_edge_representation(image_norm)
print(f"Edge map unique values: {np.unique(edges)}")  # [0., 1.]
print(f"Edge sparsity: {(edges == 0).sum() / edges.size * 100:.1f}%")  # Mostly zeros
```

**Characteristics:**
- Sparse representation (many zeros, few edges)
- Binary output (0 = non-edge, 1 = edge)
- Emphasizes geometric features
- Loses intensity information

**Theoretical Justification:**
- Hotspots correlate with geometric transitions
- Manufacturing stress concentrates at feature boundaries
- Canny thresholds (50, 150) are empirically tuned for this domain

#### Representation 3: Density Map

```python
def get_density_representation(image: np.ndarray) -> np.ndarray:
    """
    Density representation: Gaussian-smoothed density map.
    
    What it captures:
    - Local feature density/crowding
    - Spatial concentration of layout features
    - Manufacturing stress distribution (dense regions = high stress)
    
    Processing:
    1. Binarize image (threshold at 128/2 = 0.5 in normalized scale)
    2. Apply Gaussian blur (kernel 11×11, σ=5)
    3. Rescale to [0, 1]
    
    Gaussian Parameters:
    - Kernel size: 11×11 (covers ~5.5% of 1200×1200 image)
    - Sigma: 5 pixels (standard deviation)
    - Covers ~30nm² region in real lithography units
    
    Shape: (H, W) continuous density heatmap
    """
    # Binarize: features (high intensity) = 1, background = 0
    threshold = 0.5 if image.max() <= 1.0 else 128
    binary_image = (image > threshold).astype(np.float32)
    
    # Apply Gaussian blur for density smoothing
    density_raw = cv2.GaussianBlur(binary_image, ksize=(11, 11), sigmaX=5, sigmaY=5)
    
    # Normalize to [0, 1]
    density_normalized = density_raw / (density_raw.max() + 1e-8)
    
    return density_normalized

# Usage
image = load_image("HS0.png")
image_norm = normalize_image(image)
density = get_density_representation(image_norm)
print(f"Density map range: {density.min():.3f}-{density.max():.3f}")  # [0, 1]
print(f"Mean density: {density.mean():.3f}")  # Varies by benchmark
```

**Characteristics:**
- Dense representation (most pixels have values)
- Continuous output (smoother than original)
- Captures aggregated spatial information
- Loses fine-grained details

**Physical Justification:**
- Manufacturing stress is unevenly distributed
- High-density regions experience greater lithographic distortion
- Gaussian smoothing mimics stress propagation in resist chemistry

### 2.4 Combined Representation (Fusion)

```python
def get_all_representations(image: np.ndarray) -> tuple:
    """
    Generate all 3 representations.
    
    Returns:
        (original, edge, density) - each normalized to [0, 1]
    """
    original = image  # Already normalized
    edge = get_edge_representation(image)
    density = get_density_representation(image)
    return original, edge, density

def concatenate_representations(original, edge, density) -> np.ndarray:
    """
    Stack representations into 3-channel image for fusion.
    
    Args:
        original, edge, density: (H, W) arrays, normalized [0, 1]
    
    Returns:
        Concatenated: (3, H, W) array for PyTorch input
    """
    stacked = np.stack([original, edge, density], axis=0)  # (3, 1200, 1200)
    return stacked

# Usage
image = load_image("HS0.png")
image_norm = normalize_image(image)
orig, edge, dens = get_all_representations(image_norm)
fused = concatenate_representations(orig, edge, dens)
print(f"Fused representation shape: {fused.shape}")  # (3, 1200, 1200)
```

---

## Part 3: Data Splitting & Stratification

### 3.1 Why Stratification Matters

**Problem:** Random split can accidentally create imbalanced subsets
- E.g., random 80/20 split of ICCAD-1 (99 HS) might put all 99 HS in training
- Or distribution: 50 HS in train (2.5%), 49 HS in val (20.9%) — very different!

**Solution:** Stratified split preserves class distribution

### 3.2 Implementation

```python
from sklearn.model_selection import train_test_split

def stratified_split(image_paths, labels, test_size=0.2, random_seed=42):
    """
    Split data while preserving class distribution.
    
    Args:
        image_paths: List of image file paths
        labels: List of labels (0=NHS, 1=HS)
        test_size: Validation fraction (0.2 = 20%)
        random_seed: For reproducibility
    
    Returns:
        (train_paths, val_paths, train_labels, val_labels)
    """
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels  # CRITICAL: preserves class distribution
    )
    return train_paths, val_paths, train_labels, val_labels

# Verification
original_hs_ratio = sum(labels) / len(labels)
train_hs_ratio = sum(train_labels) / len(train_labels)
val_hs_ratio = sum(val_labels) / len(val_labels)

print(f"Original HS ratio: {original_hs_ratio:.1%}")
print(f"Train HS ratio: {train_hs_ratio:.1%}")
print(f"Val HS ratio: {val_hs_ratio:.1%}")
# Should all be approximately equal
```

**Example Output for ICCAD-1:**
```
Original HS ratio: 22.6%
Train HS ratio: 22.5% ← Very close!
Val HS ratio: 22.7%  ← Very close!
```

### 3.3 Data Splits by Benchmark

| Benchmark | Train Total | Val Total | Val % | HS Distribution |
|-----------|------------|----------|-------|-----------------|
| ICCAD-1   | 351        | 88       | 20.0% | 79 HS / 20 HS   |
| ICCAD-2   | 4,367      | 1,092    | 20.0% | 139 HS / 35 HS  |
| ICCAD-3   | 4,442      | 1,110    | 20.0% | 727 HS / 182 HS |
| ICCAD-4   | 3,638      | 909      | 20.0% | 76 HS / 19 HS   |
| ICCAD-5   | 2,194      | 548      | 20.0% | 21 HS / 5 HS    |

---

## Part 4: Evaluation Metrics

### 4.1 Why Balanced Accuracy?

**Problem with Naive Accuracy:**
```
Classifier: Predict ALL samples as NHS
Accuracy on ICCAD-5: 99.1% (predicting all NHS correctly)
But HS Recall: 0% (missed ALL hotspots) → USELESS!
```

**Solution: Balanced Accuracy**
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
                  = (TP/(TP+FN) + TN/(TN+FP)) / 2

For all-NHS classifier:
Sensitivity = 0 / (0 + 41) = 0%
Specificity = 19,327 / (19,327 + 0) = 100%
Balanced Acc = (0% + 100%) / 2 = 50% ← Random baseline!
```

### 4.2 Metrics Formulas

```python
def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0, 1)
        y_pred: Predicted labels (0, 1)
    
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
    
    return {
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,      # Same as recall
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

# Interpretation
metrics = {
    "balanced_accuracy": 0.8497,  # 84.97%
    "precision": 0.1450,           # 14.5% of predicted HS are correct
    "recall": 0.9779,              # 97.8% of actual HS found
    "specificity": 0.7215,         # 72.2% of actual NHS correctly rejected
}
print("Manufacturing Interpretation:")
print(f"  - Find hotspots: {metrics['recall']:.1%} (excellent)")
print(f"  - Correct predictions: {metrics['precision']:.1%} (many false alarms)")
print(f"  - Trade-off: Catch defects but expect design iteration")
```

### 4.3 Metric Importance for Manufacturing

| Metric | Manufacturing Interpretation | Priority |
|--------|------------------------------|----------|
| **Recall** | % of actual hotspots detected | **CRITICAL** |
| **Precision** | % of predicted hotspots correct | Secondary |
| **Balanced Acc** | Fair metric for imbalanced data | Reporting |
| **F1-Score** | Harmonic mean of precision/recall | Summary |

**Decision Rule:** Maximize recall, even at cost of lower precision.

---

## Part 5: Experimental Configuration

### 5.1 Training Hyperparameters

```python
# File: config.py

class Config:
    """Experiment configuration."""
    
    # ===== Paths =====
    DATA_ROOT = "/path/to/iccad-official"
    OUTPUT_DIR = "./results"
    
    # ===== Dataset =====
    BENCHMARKS = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
    IMAGE_SIZE = 1200
    TRAIN_SPLIT = 0.8  # 80% train, 20% validation
    RANDOM_SEED = 42   # For reproducibility
    
    # ===== Model =====
    NUM_CLASSES = 2  # Binary: HS (1) vs NHS (0)
    INPUT_CHANNELS = 1  # Grayscale (1 channel)
    NUM_FILTERS = 12
    KERNEL_SIZE = 3
    DROPOUT_RATE = 0.3
    
    # ===== Training =====
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0
    MOMENTUM = 0.9
    
    # ===== Optimization =====
    OPTIMIZER = "Adam"
    LOSS_FUNCTION = "CrossEntropyLoss"
    SCHEDULER = None  # Optional: ReduceLROnPlateau
    
    # ===== Early Stopping =====
    PATIENCE = 3  # Stop if val_acc doesn't improve for 3 epochs
    METRIC = "validation_accuracy"
    
    # ===== Device =====
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ===== Batch Norm =====
    BN_MOMENTUM = 0.99
    BN_EPSILON = 0.001
    
    # ===== Representations =====
    REPRESENTATIONS = ["original", "edge", "density"]
    CANNY_LOW = 50
    CANNY_HIGH = 150
    GAUSSIAN_KERNEL = 11
    GAUSSIAN_SIGMA = 5
    DENSITY_THRESHOLD = 0.5
```

### 5.2 Environmental Details

```python
# Reproducibility
import numpy as np
import torch
import random

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 5.3 Device Configuration

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Typical Resource Usage
print("\nExpected Resource Requirements:")
print("- Memory: 2-4 GB per batch")
print("- Training time: 30-45 min (GPU) / 2-3 hours (CPU)")
print("- Inference: 6.3 ms/image (GPU) / 50-100 ms/image (CPU)")
```

---

## Part 6: Troubleshooting

### Issue: "Image not found" error

```
Solution: Verify dataset extraction
$ ls /path/to/iccad-official/iccad1/train/train_hs/
# Should show: HS0.png, HS1.png, ...
```

### Issue: "Out of Memory" error

```python
# Solution 1: Reduce batch size
config.BATCH_SIZE = 16  # or 8

# Solution 2: Use CPU
config.DEVICE = "cpu"

# Solution 3: Resize images
config.IMAGE_SIZE = 512  # or 256
```

### Issue: Low validation accuracy on ICCAD-4/5

```
Expected behavior: ICCAD-4 and ICCAD-5 have insufficient training data
- ICCAD-4: 95 hotspots (very small)
- ICCAD-5: 26 hotspots (too small)

Solution: Accept lower performance or use:
- Data augmentation (rotation, elastic deformation)
- Transfer learning from ICCAD-1, 2, 3
```

---

## References

1. ICCAD-12 Benchmark: http://www.ispd.cc/contests/12/%5D(https://drive.google.com/file/d/1jx7gDR92sqoIw2Nh4NGwwZwC-qNp2osd/view)
2. Image Processing: OpenCV 4.x documentation
3. Dataset Handling: scikit-learn and PyTorch documentation
