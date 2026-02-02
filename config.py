"""
Configuration for fruit image recognition model.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2  # Used when no separate validation folder exists

# Data augmentation (OpenCV / TF)
USE_AUGMENTATION = True
RANDOM_SEED = 42

# Create required directories
for d in [DATA_DIR, TRAIN_DIR, VAL_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
