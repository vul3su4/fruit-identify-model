"""
Prediction / inference for fruit recognition.
"""
import os
import json
import numpy as np
import cv2
import tensorflow as tf
from config import IMG_HEIGHT, IMG_WIDTH, MODEL_SAVE_DIR


def load_model_and_classes(model_dir=MODEL_SAVE_DIR):
    """
    Load saved Keras model and class names.
    """
    model_path = os.path.join(model_dir, "best_fruit_model.keras")
    if not os.path.isfile(model_path):
        model_path = os.path.join(model_dir, "fruit_model_final.keras")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No model found in {model_dir}. Run train.py first."
        )
    model = tf.keras.models.load_model(model_path)
    class_names_path = os.path.join(model_dir, "class_names.json")
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_image_opencv(image_path_or_array, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Load and preprocess a single image with OpenCV.
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path_or_array}")
    else:
        img = np.asarray(image_path_or_array)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_single(model, class_names, image_path_or_array, top_k=3):
    """
    Predict fruit class for one image.
    """
    x = preprocess_image_opencv(image_path_or_array)
    logits = model.predict(x, verbose=0)[0]
    top_indices = np.argsort(logits)[::-1][:top_k]
    results = [
        (class_names[i], float(logits[i]))
        for i in top_indices
    ]
    return results


def predict_and_display(image_path, model_dir=MODEL_SAVE_DIR):
    """
    Load model, predict on image, print top predictions.
    """
    model, class_names = load_model_and_classes(model_dir)
    results = predict_single(model, class_names, image_path)
    print("Predictions:")
    for name, prob in results:
        print(f"  {name}: {prob:.4f}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict_and_display(sys.argv[1])
