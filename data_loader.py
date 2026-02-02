"""
Data loading and preprocessing with OpenCV + TensorFlow.
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from config import (
    TRAIN_DIR,
    VAL_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)


def load_image_with_opencv(path: str, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Use OpenCV to load and preprocess a single image.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img


def get_class_names_from_dir(data_dir: str):
    """
    Get sorted class names from directory structure: data_dir/class_name/...
    """
    if not os.path.isdir(data_dir):
        return []
    names = [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ]
    return sorted(names)


def build_tf_dataset_from_folders(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
):
    """
    Build TensorFlow datasets from folder structure.
    Expects: train_dir/class_name/*.jpg, val_dir/class_name/*.jpg
    If val_dir has no subdirs, will split from train_dir using validation_split.
    """
    class_names = get_class_names_from_dir(train_dir)
    if not class_names:
        raise FileNotFoundError(
            f"No class subdirectories found in {train_dir}. "
            "Please add folders like: train/apple/, train/banana/, ..."
        )

    # Use separate validation folder if it has the same class subdirs
    if os.path.isdir(val_dir) and get_class_names_from_dir(val_dir):
        train_ds = image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            color_mode="rgb",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=RANDOM_SEED,
            interpolation="bilinear",
        )
        val_ds = image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            color_mode="rgb",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            interpolation="bilinear",
        )
    else:
        full_ds = image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            color_mode="rgb",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=RANDOM_SEED,
            validation_split=validation_split,
            subset="both",
            interpolation="bilinear",
        )
        train_ds, val_ds = full_ds

    # Normalize to [0, 1] (Rescaling can also be done in the model)
    normalize = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def augment_with_tf(image, label):
    """
    Data augmentation in TensorFlow graph (optional, for use in dataset.map).
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label
