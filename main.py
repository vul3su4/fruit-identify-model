"""
Fruit image recognition - main entry.
TensorFlow + OpenCV training pipeline.
"""
import os
import argparse
import tensorflow as tf
import cv2

from config import TRAIN_DIR, VAL_DIR
from data_loader import build_tf_dataset_from_folders, get_class_names_from_dir
from model import build_fruit_cnn
from train import train
from predict import load_model_and_classes, predict_and_display


def cmd_train(args):
    """Run training."""
    train(
        train_dir=args.train_dir or TRAIN_DIR,
        val_dir=args.val_dir or VAL_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def cmd_predict(args):
    """Run prediction on an image."""
    from config import MODEL_SAVE_DIR
    predict_and_display(args.image, model_dir=args.model_dir or MODEL_SAVE_DIR)


def cmd_info(args):
    """Show environment and data info."""
    print("TensorFlow version:", tf.__version__)
    print("OpenCV version:", cv2.__version__)
    print("Train dir:", TRAIN_DIR)
    print("Validation dir:", VAL_DIR)
    classes = get_class_names_from_dir(TRAIN_DIR)
    print("Classes found in train:", classes if classes else "(none - add data/train/<class_name>/)")


def main():
    parser = argparse.ArgumentParser(
        description="Fruit image recognition (TensorFlow + OpenCV)"
    )
    sub = parser.add_subparsers(dest="command", help="command")

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--train-dir", default=None, help="Training data directory")
    p_train.add_argument("--val-dir", default=None, help="Validation data directory")
    p_train.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_predict = sub.add_parser("predict", help="Predict fruit from image")
    p_predict.add_argument("image", help="Path to image file")
    p_predict.add_argument("--model-dir", default=None, help="Directory with saved model")
    p_predict.set_defaults(func=cmd_predict)

    # info
    p_info = sub.add_parser("info", help="Show environment and data info")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
