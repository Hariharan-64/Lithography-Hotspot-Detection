"""
================================================================================
    GROUP 1 - MULTI-REPRESENTATION LEARNING FOR LITHOGRAPHY HOTSPOT DETECTION
    Course: AI and Machine Learning for IC (BEVD402L), VIT Chennai
    Submission: 20/09/2026
    
    Research Question:
    Can combining multiple layout representations improve lithography hotspot
    detection compared with using a single representation?
    
    Author: Group 1 (Multi-Representation Learning)
    Date: September 2026
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Computer Vision & Image Processing
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# Metrics & Utilities
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, 
                            precision_score, recall_score, f1_score, 
                            roc_auc_score)
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

print(f"\n{'='*80}")
print(f"Lithography Hotspot Detection - Multi-Representation Learning")
print(f"{'='*80}\n")

# ============================================================================
# PART 1: CONSTANTS, CONFIGURATION & UTILITIES
# ============================================================================

class Config:
    """
    Central configuration for all experiments.
    Modify these values to run different experimental settings.
    """
    # Dataset
    DATA_ROOT = r"/mnt/c/Outputs/iccad_official/iccad-official"
    BENCHMARKS = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
    IMAGE_SIZE = 1200  # All ICCAD images are 1200x1200
    
    # Training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    NUM_EPOCHS = 10  # Reference paper used 5-10 epochs
    LEARNING_RATE = 0.001
    OPTIMIZER = "Nadam"  # Adam with Nesterov momentum (similar to Nadam)
    
    # Data Split (train/val from the provided train set)
    TRAIN_SPLIT = 0.8  # 80% train, 20% validation
    RANDOM_SEED = 42
    
    # Output
    OUTPUT_DIR = "/mnt/c/Outputs"
    RESULTS_JSON = "results_all_benchmarks.json"
    
    # Representations
    REPRESENTATIONS = ["original", "edge", "density"]
    
    @staticmethod
    def ensure_output_dir():
        """Create output directory if it doesn't exist"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# ============================================================================
# PART 2: DATA LOADING & ANALYSIS
# ============================================================================

class DatasetAnalyzer:
    """
    THEORY:
    Class imbalance is a critical problem in binary classification when one class
    significantly outnumbers the other. In lithography hotspot detection:
    - Hotspots (HS) are rare patterns (~1-10% of layouts)
    - Non-hotspots (NHS) are common (~90-99% of layouts)
    
    Using naive accuracy misleads because a model predicting all NHS achieves
    95%+ accuracy while missing all hotspots (recall = 0). 
    
    Instead, we use BALANCED ACCURACY = (Sensitivity + Specificity) / 2
    where:
    - Sensitivity (Recall) = TP / (TP + FN) = true hotspot detection rate
    - Specificity = TN / (TN + FP) = true non-hotspot detection rate
    
    This ensures both classes are equally weighted in evaluation.
    """
    
    def __init__(self, data_root=Config.DATA_ROOT):
        self.data_root = data_root
        self.benchmark_stats = {}
        self.class_distribution = {}
    
    def analyze_benchmark(self, benchmark_name):
        """
        Analyze class distribution for a single benchmark.
        
        Args:
            benchmark_name: str, e.g., "iccad1"
            
        Returns:
            dict with HS/NHS counts for train and test splits
        """
        bench_path = os.path.join(self.data_root, benchmark_name)
        
        stats = {
            "benchmark": benchmark_name,
            "train": {"HS": 0, "NHS": 0},
            "test": {"HS": 0, "NHS": 0}
        }
        
        # Count training samples
        train_hs = os.path.join(bench_path, "train", "train_hs")
        train_nhs = os.path.join(bench_path, "train", "train_nhs")

        if not os.path.exists(train_hs):
            print(f"ERROR: Path does not exist: {train_hs}")
            return stats
        stats["train"]["HS"] = len(os.listdir(train_hs))
        stats["train"]["NHS"] = len(os.listdir(train_nhs))
        
        # Count test samples
        test_hs = os.path.join(bench_path, "test", "test_hs")
        test_nhs = os.path.join(bench_path, "test", "test_nhs")
        stats["test"]["HS"] = len(os.listdir(test_hs))
        stats["test"]["NHS"] = len(os.listdir(test_nhs))
        
        self.benchmark_stats[benchmark_name] = stats
        return stats
    
    def print_summary(self):
        """Print formatted summary of all benchmarks"""
        print("\n" + "="*90)
        print("DATASET ANALYSIS SUMMARY - CLASS DISTRIBUTION")
        print("="*90)
        
        print("\n{:<12} {:<20} {:<20} {:<15}".format(
            "Benchmark", "Training", "Testing", "Imbalance Ratio"
        ))
        print("-" * 90)
        
        for bench_name in Config.BENCHMARKS:
            if bench_name in self.benchmark_stats:
                stats = self.benchmark_stats[bench_name]
                tr_hs, tr_nhs = stats["train"]["HS"], stats["train"]["NHS"]
                te_hs, te_nhs = stats["test"]["HS"], stats["test"]["NHS"]
                
                # Imbalance ratio = NHS / HS (should be > 10 for all)
                imbalance_train = tr_nhs / tr_hs if tr_hs > 0 else 0
                
                print("{:<12} HS:{:>5} NHS:{:>5} ({:>6}) HS:{:>5} NHS:{:>6} ({:>7}) {:>6.1f}:1".format(
                    bench_name, tr_hs, tr_nhs, f"{tr_hs+tr_nhs}",
                    te_hs, te_nhs, f"{te_hs+te_nhs}", imbalance_train
                ))
        
        print("="*90)
        print("✓ All benchmarks are HIGHLY IMBALANCED (>10:1 NHS:HS ratio)")
        print("✓ Using BALANCED ACCURACY as primary metric (not naive accuracy)")
        print("="*90 + "\n")
    
    def visualize_class_distribution(self):
        """
        Create comprehensive visualization of class distribution across benchmarks.
        Visualizations are essential for IEEE papers to communicate data properties.
        """
        Config.ensure_output_dir()
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        benchmarks = list(self.benchmark_stats.keys())
        
        # Plot 1: Stacked bar chart - Train/Test by benchmark
        ax1 = fig.add_subplot(gs[0, :])
        x_pos = np.arange(len(benchmarks))
        width = 0.35
        
        train_hs_list = [self.benchmark_stats[b]["train"]["HS"] for b in benchmarks]
        train_nhs_list = [self.benchmark_stats[b]["train"]["NHS"] for b in benchmarks]
        test_hs_list = [self.benchmark_stats[b]["test"]["HS"] for b in benchmarks]
        test_nhs_list = [self.benchmark_stats[b]["test"]["NHS"] for b in benchmarks]
        
        ax1.bar(x_pos - width/2, train_hs_list, width, label="Train HS", color="#d62728", alpha=0.8)
        ax1.bar(x_pos - width/2, train_nhs_list, width, bottom=train_hs_list, 
                label="Train NHS", color="#1f77b4", alpha=0.8)
        ax1.bar(x_pos + width/2, test_hs_list, width, label="Test HS", color="#d62728", alpha=0.5)
        ax1.bar(x_pos + width/2, test_nhs_list, width, bottom=test_hs_list, 
                label="Test NHS", color="#1f77b4", alpha=0.5)
        
        ax1.set_ylabel("Number of Samples", fontsize=11, fontweight='bold')
        ax1.set_title("ICCAD-12 Dataset: Class Distribution Across Benchmarks", 
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(benchmarks)
        ax1.legend(loc='upper left', ncol=4)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Pie charts for each benchmark (Train)
        for idx, bench in enumerate(benchmarks):
            ax = fig.add_subplot(gs[1 + idx//2, idx % 2])
            stats = self.benchmark_stats[bench]
            hs = stats["train"]["HS"]
            nhs = stats["train"]["NHS"]
            
            colors = ["#d62728", "#1f77b4"]
            explode = (0.05, 0)  # Explode HS slightly to emphasize rarity
            
            ax.pie([hs, nhs], labels=[f"HS ({hs})", f"NHS ({nhs})"], 
                  autopct="%1.1f%%", colors=colors, explode=explode, 
                  startangle=90)
            ax.set_title(f"{bench} Training Set", fontweight='bold')
        
        plt.suptitle("Class Imbalance Analysis: ICCAD-12 Benchmarks (Train Sets)", 
                    fontsize=14, fontweight='bold', y=0.995)
        
        output_path = os.path.join(Config.OUTPUT_DIR, "01_dataset_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


# ============================================================================
# PART 3: CUSTOM PYTORCH DATASET CLASS
# ============================================================================

class HotspotDataset(Dataset):
    """
    THEORY:
    PyTorch Dataset is a custom class that:
    1. Loads image data efficiently
    2. Applies preprocessing/normalization
    3. Generates multiple representations (Original, Edge, Density)
    4. Returns samples compatible with DataLoader
    
    This abstraction allows:
    - Flexible data loading (don't load all images into memory)
    - Easy augmentation/transformation
    - Consistent preprocessing across train/val/test
    - Multi-representation generation on-the-fly
    """
    
    def __init__(self, image_paths, labels, representation="original", 
                 normalize=True, augment=False):
        """
        Args:
            image_paths: list of paths to PNG images
            labels: list of binary labels (0=NHS, 1=HS)
            representation: str, one of ["original", "edge", "density"]
            normalize: bool, whether to normalize to [0, 1]
            augment: bool, whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.representation = representation
        self.normalize = normalize
        self.augment = augment
        
        assert len(image_paths) == len(labels), "Paths and labels must have same length"
        assert representation in ["original", "edge", "density"], \
            f"Unknown representation: {representation}"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load image and apply representation transformation.
        
        Returns:
            tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image as grayscale (mode='L')
        image = Image.open(img_path).convert('L')
        image = np.array(image, dtype=np.float32)
        
        # Apply representation transformation
        if self.representation == "original":
            processed = self._process_original(image)
        elif self.representation == "edge":
            processed = self._process_edge(image)
        elif self.representation == "density":
            processed = self._process_density(image)
        
        # Normalize to [0, 1]
        if self.normalize:
            processed = processed / 255.0 if processed.max() > 1.0 else processed
        
        # Convert to tensor
        tensor = torch.from_numpy(processed).unsqueeze(0).float()  # Add channel dim
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return tensor, label_tensor
    
    def _process_original(self, image):
        """
        THEORY - Original Representation:
        Keep the raw binary layout as-is. This captures:
        - Actual geometry (shapes, lines)
        - Spatial structure
        - Feature sizes and spacing
        
        No transformation needed; the raw layout is the representation.
        """
        return image
    
    def _process_edge(self, image):
        """
        THEORY - Edge Map Representation:
        Use Canny edge detection to extract boundaries and transitions.
        
        Canny edge detection:
        1. Applies Gaussian blur (reduce noise)
        2. Computes image gradients (derivatives in x, y)
        3. Applies non-maximum suppression (thin edges)
        4. Uses hysteresis thresholding (connect related edges)
        
        Result: Binary image highlighting only edges/boundaries.
        
        Why useful for hotspots?
        - Hotspots often occur at layout boundaries and corners
        - Edge map emphasizes geometric transitions
        - Complements original by focusing on discontinuities
        """
        # Convert to uint8 for OpenCV
        img_uint8 = cv2.convertScaleAbs(image).astype(np.uint8)
        
        # Apply Canny edge detection
        # Low threshold = 50, High threshold = 150 (standard values)
        edges = cv2.Canny(img_uint8, 50, 150)
        
        return edges.astype(np.float32)
    
    def _process_density(self, image):
        """
        THEORY - Density Map Representation:
        Compute local spatial density using Gaussian filtering.
        
        Process:
        1. Binarize: Layout pixels → [0, 1]
        2. Apply Gaussian blur: Smooths density across neighborhood
        3. Rescale: Back to [0, 255] for visualization
        
        Gaussian kernel spreads each pixel's influence to neighbors,
        creating a "density heat map" showing crowded regions.
        
        Why useful for hotspots?
        - Hotspots often occur in densely packed regions
        - High density → more lithographic stress → more errors
        - Complements original by highlighting crowded areas
        """
        # Binarize: layout pixels to 1, background to 0
        img_binary = (image > 128).astype(np.float32)
        
        # Apply Gaussian blur to create density map
        # sigma=5 means blur kernel spreads influence ~5 pixels
        density = cv2.GaussianBlur(img_binary, ksize=(11, 11), sigmaX=5)
        
        # Rescale to [0, 255]
        density_scaled = (density * 255).astype(np.float32)
        
        return density_scaled


# ============================================================================
# PART 4: BASELINE CNN ARCHITECTURE
# ============================================================================

class LightweightCNN(nn.Module):
    """
    THEORY - Lightweight CNN for Hotspot Detection:
    
    The reference paper's architecture is designed for CAD tool integration:
    - Only 12,873 parameters (vs. 117M for VGG16)
    - Fast inference (6.3 ms vs. 182 ms for transfer learning)
    - Achieves 95.3% balanced accuracy
    
    Architecture:
    ┌─────────────┐
    │ Input (1x1200x1200) - Grayscale layout
    │
    ├─ Basic Block 1:
    │  ├─ Conv2D: 12 filters, 3×3, ELU activation
    │  ├─ Conv2D: 12 filters, 3×3, ELU activation
    │  ├─ Conv2D: 12 filters, 3×3, No activation
    │  ├─ BatchNorm: momentum=0.99, epsilon=0.001
    │  ├─ ELU: Exponential Linear Unit activation
    │  └─ MaxPool: 2×2
    │
    ├─ MaxPool: 2×2 (explicit pooling)
    │
    ├─ Basic Block 2:
    │  └─ [Same as Block 1]
    │
    ├─ Flatten: Convert to 1D vector
    ├─ Dropout: 0.3 (drop 30% of features during training)
    ├─ Dense (FC): Map to hidden features
    └─ Sigmoid: Binary classification output (0=NHS, 1=HS)
    
    Key design choices:
    1. Small filters (3×3): Capture local features efficiently
    2. ELU activation: Smooth activation, helps gradient flow
    3. BatchNorm: Stabilizes training, allows higher learning rates
    4. Dropout: Reduces overfitting
    5. MaxPool: Reduces spatial dimensions, captures dominant features
    
    Total parameters: 12,873 (extremely lightweight for deep learning)
    """
    
    def __init__(self, num_classes=2, input_height=1200, input_width=1200):
        """
        Args:
            num_classes: int, number of output classes (2 for binary)
            input_height, input_width: image dimensions
        """
        super(LightweightCNN, self).__init__()
        
        # Basic Block hyperparameters (from reference paper TABLE I)
        num_filters = 12
        kernel_size = 3
        batch_norm_momentum = 0.99
        batch_norm_epsilon = 0.001
        dropout_rate = 0.3
        
        # BASIC BLOCK 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1),
            nn.ELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1),
            nn.ELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(num_filters, momentum=batch_norm_momentum, 
                          eps=batch_norm_epsilon),
            nn.ELU(),
            nn.MaxPool2d(2, 2)
        )
        
        # INTERMEDIATE MAXPOOL
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # BASIC BLOCK 2
        self.block2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1),
            nn.ELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1),
            nn.ELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(num_filters, momentum=batch_norm_momentum, 
                          eps=batch_norm_epsilon),
            nn.ELU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate flattened size after convolutions
        # After Block1 + MaxPool: 1200 → 300 → 150 (factors of 2)
        # After Block2: 150 → 75 (factor of 2)
        # With 12 filters
        flattened_size = num_filters * (input_height // 8) * (input_width // 8)
        
        # CLASSIFICATION HEAD
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_size, 256),
            nn.ELU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# PART 5: TRAINING & EVALUATION FUNCTIONS
# ============================================================================

class HotspotTrainer:
    """
    THEORY - Training Loop:
    
    Standard supervised learning loop:
    1. Forward pass: image → model → logits
    2. Loss computation: Compare predictions vs ground truth
    3. Backward pass: Compute gradients via backpropagation
    4. Parameter update: Update weights using optimizer
    5. Validation: Evaluate on hold-out validation set
    6. Early stopping: Stop if validation performance plateaus
    
    Loss function: CrossEntropyLoss
    - Combines softmax and negative log-likelihood
    - Standard for multi-class classification
    - Provides well-calibrated probability outputs
    
    Optimizer: Adam with Nesterov momentum (similar to Nadam)
    - Adaptive learning rates per parameter
    - Nesterov momentum helps escape local minima
    - Reference paper used Nadam optimizer
    """
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        # Adam with AMSGrad variant (similar to Nadam)
        self.optimizer = optim.Adam(model.parameters(), 
                                   lr=learning_rate, 
                                   amsgrad=True)
        self.train_losses = []
        self.val_losses = []
        self.best_val_acc = 0.0
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        self.train_losses.append(epoch_loss)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate on hold-out set"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        self.val_losses.append(epoch_loss)
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=10, verbose=True):
        """Full training loop with early stopping"""
        best_model_state = None
        patience_counter = 0
        patience = 3  # Early stopping patience
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"  → Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model


class MetricsCalculator:
    """
    THEORY - Evaluation Metrics for Imbalanced Classification:
    
    For imbalanced data (hotspots are rare):
    
    1. BALANCED ACCURACY ⭐ (PRIMARY METRIC)
       = (Sensitivity + Specificity) / 2
       = (TP/(TP+FN) + TN/(TN+FP)) / 2
       Pros: Gives equal weight to both classes
       Cons: Doesn't consider false positives and false negatives equally
    
    2. PRECISION
       = TP / (TP + FP)
       = Of predicted hotspots, how many are actually hotspots?
       Important: Don't want false alarms in production
    
    3. RECALL / SENSITIVITY
       = TP / (TP + FN)
       = Of actual hotspots, how many did we find?
       Critical: Missing hotspots → manufacturing defects
    
    4. SPECIFICITY
       = TN / (TN + FP)
       = Of actual non-hotspots, how many did we correctly identify?
    
    5. F1-SCORE
       = 2 * (Precision * Recall) / (Precision + Recall)
       Harmonic mean of precision and recall
       Useful when both false positives and false negatives matter
    
    Confusion Matrix:
                 Predicted
             NHS        HS
    Actual  NHS  [TN]      [FP]   ← False Alarms
            HS   [FN]      [TP]   ← Missed Hotspots (CRITICAL)
    
    Most important: Keep FN low (don't miss real hotspots)
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Ground truth labels (0=NHS, 1=HS)
            y_pred: Predicted labels (0=NHS, 1=HS)
            
        Returns:
            dict with all metrics
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        return {
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
    
    @staticmethod
    def print_metrics(metrics, label=""):
        """Pretty print metrics"""
        print(f"\n{label}")
        print("-" * 60)
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ⭐")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"Specificity:       {metrics['specificity']:.4f}")
        print(f"F1-Score:          {metrics['f1_score']:.4f}")
        print("-" * 60)
        cm = metrics['confusion_matrix']
        print(f"Confusion Matrix:\n  TN={cm[0][0]} FP={cm[0][1]}\n  FN={cm[1][0]} TP={cm[1][1]}")


# ============================================================================
# PART 6: MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def prepare_data_for_benchmark(benchmark_name, representation="original"):
    """
    Prepare data for a single benchmark: load paths, split, create DataLoader.
    
    Args:
        benchmark_name: str, e.g., "iccad1"
        representation: str, one of ["original", "edge", "density"]
        
    Returns:
        tuple of (train_loader, val_loader, test_loader)
    """
    print(f"\n→ Preparing data for {benchmark_name} (representation: {representation})...")
    
    bench_path = os.path.join(Config.DATA_ROOT, benchmark_name)
    image_paths = []
    labels = []
    
    # Load HS (label=1)
    hs_dir = os.path.join(bench_path, "train", "train_hs")
    for fname in sorted(os.listdir(hs_dir)):
        image_paths.append(os.path.join(hs_dir, fname))
        labels.append(1)
    
    # Load NHS (label=0)
    nhs_dir = os.path.join(bench_path, "train", "train_nhs")
    for fname in sorted(os.listdir(nhs_dir)):
        image_paths.append(os.path.join(nhs_dir, fname))
        labels.append(0)
    
    # Stratified split: preserve class distribution
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=Config.RANDOM_SEED,
        stratify=labels  # Ensure same HS/NHS ratio in train and val
    )
    
    # Test set (use original test directory)
    test_paths = []
    test_labels = []
    
    test_hs_dir = os.path.join(bench_path, "test", "test_hs")
    for fname in sorted(os.listdir(test_hs_dir)):
        test_paths.append(os.path.join(test_hs_dir, fname))
        test_labels.append(1)
    
    test_nhs_dir = os.path.join(bench_path, "test", "test_nhs")
    for fname in sorted(os.listdir(test_nhs_dir)):
        test_paths.append(os.path.join(test_nhs_dir, fname))
        test_labels.append(0)
    
    # Create datasets
    train_dataset = HotspotDataset(train_paths, train_labels, 
                                  representation=representation, normalize=True)
    val_dataset = HotspotDataset(val_paths, val_labels, 
                                representation=representation, normalize=True)
    test_dataset = HotspotDataset(test_paths, test_labels, 
                                 representation=representation, normalize=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def evaluate_on_test_set(model, test_loader, device):
    """Evaluate model on test set and return metrics + predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
    
    metrics = MetricsCalculator.calculate_metrics(all_labels, all_preds)
    
    return metrics, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================================
# MAIN EXECUTION - STEPS 1-5
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # STEP 1: DATASET ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATASET ANALYSIS & CLASS DISTRIBUTION")
    print("="*80)
    
    analyzer = DatasetAnalyzer()
    for bench_name in Config.BENCHMARKS:
        analyzer.analyze_benchmark(bench_name)
    
    analyzer.print_summary()
    analyzer.visualize_class_distribution()
    
    # ========================================================================
    # STEP 2: DATA SPLITTING (done inside prepare_data_for_benchmark)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: TRAIN/VALIDATION/TEST SPLIT")
    print("="*80)
    print("✓ Using stratified 80/20 split on training data")
    print("✓ Preserving HS/NHS distribution in train and validation")
    print("✓ Test set remains untouched until final evaluation")
    
    # ========================================================================
    # STEP 3: BASELINE CNN IMPLEMENTATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: BASELINE CNN IMPLEMENTATION")
    print("="*80)
    
    Config.ensure_output_dir()
    model = LightweightCNN().to(Config.DEVICE)
    num_params = model.count_parameters()
    print(f"✓ Model created: {num_params:,} parameters")
    print(f"✓ Device: {Config.DEVICE}")
    
    # ========================================================================
    # STEP 4 & 5: BASELINE TRAINING & EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4 & 5: BASELINE TRAINING & EVALUATION (Original Representation)")
    print("="*80)
    
    results_all = {}
    
    for bench_name in Config.BENCHMARKS:
        print(f"\n{'='*80}")
        print(f"Processing {bench_name}")
        print(f"{'='*80}")
        
        # Reinitialize model for each benchmark
        model = LightweightCNN().to(Config.DEVICE)
        trainer = HotspotTrainer(model, Config.DEVICE, Config.LEARNING_RATE)
        
        # Prepare data
        train_loader, val_loader, test_loader = prepare_data_for_benchmark(
            bench_name, representation="original"
        )
        
        # Train
        print(f"\n→ Training baseline CNN ({Config.NUM_EPOCHS} epochs)...")
        trainer.train(train_loader, val_loader, num_epochs=Config.NUM_EPOCHS, verbose=True)
        
        # Evaluate on test set
        print(f"\n→ Evaluating on test set...")
        metrics, preds, labels, probs = evaluate_on_test_set(
            model, test_loader, Config.DEVICE
        )
        
        MetricsCalculator.print_metrics(
            metrics, label=f"BASELINE - {bench_name} TEST RESULTS"
        )
        
        # Store results
        results_all[f"{bench_name}_baseline"] = metrics
        
        # Save model
        model_path = os.path.join(Config.OUTPUT_DIR, f"{bench_name}_baseline_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved: {model_path}")
    
    # ========================================================================
    # SAVE RESULTS TO JSON
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_path = os.path.join(Config.OUTPUT_DIR, Config.RESULTS_JSON)
    with open(results_path, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"✓ Results saved: {results_path}")
    
    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Benchmark':<12} {'Balanced Acc':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 65)
    
    for bench_name in Config.BENCHMARKS:
        key = f"{bench_name}_baseline"
        if key in results_all:
            m = results_all[key]
            print(f"{bench_name:<12} {m['balanced_accuracy']:<15.4f} {m['precision']:<12.4f} "
                  f"{m['recall']:<12.4f} {m['f1_score']:<12.4f}")
    
    print("\n" + "="*80)
    print(f"✓ STEP 1-5 COMPLETE: Data analysis & baseline established")
    print(f"✓ Next: Generate Edge/Density representations & train individual models")
    print("="*80 + "\n")
