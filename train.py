"""
Training script for fruit recognition model.
"""
import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_loader import build_tf_dataset_from_folders, augment_with_tf
from model import build_fruit_cnn
from config import (
    TRAIN_DIR,
    VAL_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_DIR,
    LOG_DIR,
    USE_AUGMENTATION,
)


def train(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    model_save_dir=MODEL_SAVE_DIR,
    use_augmentation=USE_AUGMENTATION,
):
    """
    Load data, build model, train and save.
    """
    # Build datasets
    train_ds, val_ds, class_names = build_tf_dataset_from_folders(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=image_size,
        batch_size=batch_size,
    )
    num_classes = len(class_names)

    if use_augmentation:
        train_ds = train_ds.map(
            augment_with_tf,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_fruit_cnn(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    os.makedirs(model_save_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_save_dir, "best_fruit_model.keras")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Save class names for prediction
    class_names_path = os.path.join(model_save_dir, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save final weights (optional)
    final_path = os.path.join(model_save_dir, "fruit_model_final.keras")
    model.save(final_path)
    print(f"Model saved to {checkpoint_path} (best) and {final_path} (final).")
    return history, model, class_names


if __name__ == "__main__":
    train()
