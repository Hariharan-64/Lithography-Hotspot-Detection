"""
Configuration File for Lithography Hotspot Detection Project
Group 1 - AI and Machine Learning for IC (BEVD402L)
VIT Chennai
"""

import torch
import os

class Config:
    """Central configuration for all experiments."""
    
    # ============================================================================
    # PATHS
    # ============================================================================
    
    # Update this to your actual dataset location
    DATA_ROOT = "/path/to/iccad-official"
    
    # Output directory for results
    OUTPUT_DIR = "./results"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    # ============================================================================
    # DATASET
    # ============================================================================
    
    BENCHMARKS = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
    IMAGE_SIZE = 1200
    NUM_CLASSES = 2  # Binary: HS (1) vs NHS (0)
    TRAIN_SPLIT = 0.8  # 80% train, 20% validation
    RANDOM_SEED = 42
    
    
    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    
    INPUT_CHANNELS = 1  # Grayscale images
    NUM_FILTERS = 12
    KERNEL_SIZE = 3
    DROPOUT_RATE = 0.3
    BN_MOMENTUM = 0.99
    BN_EPSILON = 0.001
    
    
    # ============================================================================
    # TRAINING
    # ============================================================================
    
    BATCH_SIZE = 32  # Reduce to 16 or 8 if out of memory
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    
    # ============================================================================
    # OPTIMIZATION
    # ============================================================================
    
    OPTIMIZER = "Adam"
    LOSS_FUNCTION = "CrossEntropyLoss"
    
    
    # ============================================================================
    # EARLY STOPPING
    # ============================================================================
    
    PATIENCE = 3  # Stop if validation accuracy doesn't improve for 3 epochs
    METRIC = "validation_accuracy"
    
    
    # ============================================================================
    # DEVICE
    # ============================================================================
    
    # Use GPU if available, else CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Force CPU (uncomment to disable GPU):
    # DEVICE = "cpu"
    
    
    # ============================================================================
    # REPRESENTATIONS
    # ============================================================================
    
    REPRESENTATIONS = ["original", "edge", "density"]
    
    # Canny edge detection parameters
    CANNY_LOW = 50
    CANNY_HIGH = 150
    
    # Gaussian blur parameters for density map
    GAUSSIAN_KERNEL = 11
    GAUSSIAN_SIGMA = 5
    DENSITY_THRESHOLD = 0.5
    
    
    # ============================================================================
    # DISPLAY CONFIGURATION
    # ============================================================================
    
    @staticmethod
    def print_config():
        """Print all configuration settings."""
        print("="*80)
        print("CONFIGURATION SETTINGS")
        print("="*80)
        print(f"DATA_ROOT: {Config.DATA_ROOT}")
        print(f"OUTPUT_DIR: {Config.OUTPUT_DIR}")
        print(f"BENCHMARKS: {Config.BENCHMARKS}")
        print(f"BATCH_SIZE: {Config.BATCH_SIZE}")
        print(f"NUM_EPOCHS: {Config.NUM_EPOCHS}")
        print(f"LEARNING_RATE: {Config.LEARNING_RATE}")
        print(f"DEVICE: {Config.DEVICE}")
        print(f"REPRESENTATIONS: {Config.REPRESENTATIONS}")
        print("="*80)


if __name__ == "__main__":
    Config.print_config()
