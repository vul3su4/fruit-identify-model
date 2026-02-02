# Fruit Image Recognition (TensorFlow + OpenCV)

Training pipeline for fruit image recognition using TensorFlow and OpenCV.

## Project structure

```
fruit-identify-model/
├── config.py        # Paths and hyperparameters
├── data_loader.py   # Data loading and preprocessing (OpenCV + TF Dataset)
├── model.py         # CNN model architecture
├── train.py         # Training script
├── predict.py       # Prediction / inference
├── main.py          # Main entry (train / predict / info)
├── data/            # Put training data here
│   ├── train/       # Training set, one folder per class
│   │   ├── apple/
│   │   ├── banana/
│   │   └── ...
│   └── validation/  # Validation set (optional, same structure as train)
├── saved_models/    # Trained model and class_names.json
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Data preparation

1. Under `data/train/`, create one folder per fruit class, e.g.:
   - `data/train/apple/` — put apple images
   - `data/train/banana/` — put banana images
2. If you have a validation set, create the same class folders under `data/validation/`.

## Usage

- **Show environment and data info**  
  `python main.py info`

- **Train the model**  
  `python main.py train`  
  Optional: `--epochs 20 --batch-size 32 --train-dir ... --val-dir ...`

- **Predict on a single image**  
  `python main.py predict path/to/image.jpg`  
  Optional: `--model-dir saved_models`

- **Train only (without main)**  
  `python train.py`

- **Predict only**  
  `python predict.py path/to/image.jpg`

## Architecture overview

- **OpenCV**: Single-image load, resize, BGR↔RGB, preprocessing (in `data_loader.py` and `predict.py`).
- **TensorFlow/Keras**: `image_dataset_from_directory` for Dataset, Rescaling, optional `augment_with_tf`; CNN training, save/load.
- **Model**: `build_fruit_cnn` in `model.py` is a multi-layer CNN (Conv2D + BatchNorm + Pooling + Dense) with softmax output.

After training, the best model is saved as `saved_models/best_fruit_model.keras` and class names as `saved_models/class_names.json`; prediction loads these automatically.
