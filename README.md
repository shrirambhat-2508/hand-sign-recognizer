# Hand Sign Recognizer 🤟

A real-time American Sign Language (ASL) letter recognition system built with a Convolutional Neural Network (CNN) and MediaPipe hand landmark detection.

## Project Structure

```
hand-sign-recognizer/
├── train_model.py        # Train the CNN on Sign Language MNIST dataset
├── live_demo.py          # Real-time webcam ASL recognition using MediaPipe
├── .gitignore
└── README.md
```

## How It Works

1. **Training (`train_model.py`)** — Trains a CNN on the Sign Language MNIST dataset (28×28 grayscale images of 24 ASL letters, A–Y excluding J and Z). Outputs training curves, confusion matrix, and sample predictions to `test_outputs/`.

2. **Live Demo (`live_demo.py`)** — Uses MediaPipe to detect hand landmarks from your webcam in real time. The detected hand region is cropped, preprocessed, and fed into the trained CNN. An 8-frame temporal consensus buffer prevents flickering, and dual-orientation rechecking improves accuracy. Session outputs are saved to `live_outputs/`.

## Setup

### 1. Install dependencies

```bash
pip install tensorflow mediapipe opencv-python scikit-learn matplotlib pandas numpy
```

### 2. Dataset

The Sign Language MNIST CSV files and reference images are included in the repository:
```
Sign Language Dataset/sign_mnist_train.csv
Sign Language Dataset/sign_mnist_test.csv
```
The dataset is also available from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

### 3. Download the MediaPipe Hand Landmarker model

```python
import urllib.request
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "hand_landmarker.task"
)
```

### 4. Train the model

```bash
python train_model.py
```

Outputs saved to `test_outputs/`:
- `training_curves.png` — Accuracy & loss curves with overfitting diagnosis
- `confusion_matrix.png` — Full 24-class confusion matrix
- `confusion_matrix_2x2.png` — Simplified 2×2 confusion matrix
- `confusion_matrix_table.png` — Metrics table (Precision, Recall, F1)
- `confusion_matrix_metrics.csv` — Per-class metrics CSV
- `sample_predictions.png` — Sample test image predictions

The trained model is saved as `sign_language_cnn.h5` in the project root.

### 5. Run the live demo

```bash
python live_demo.py
```

| Key | Action |
|-----|--------|
| `Q` | Quit and save session outputs to `live_outputs/` |

**Visual indicators:**
- 🔴 Red — No hand detected
- 🟡 Amber — Rechecking confidence
- 🟢 Green — Letter verified

## Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Dense (output) | 24 units, Softmax |

**Test accuracy: ~93.8%**

## ASL Letters Supported

A B C D E F G H I K L M N O P Q R S T U V W X Y  
*(J and Z excluded — they require motion)*

## Notes

- `sign_language_cnn.h5` and `hand_landmarker.task` are **not** tracked in git (too large / auto-generated).
- A local backup of key source files is kept in `backup_working/` (also excluded from git).
