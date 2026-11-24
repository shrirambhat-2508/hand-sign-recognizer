from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import RandomRotation, RandomZoom


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR_CANDIDATES = (
    BASE_DIR / "sign_mnist_data",
    BASE_DIR / "Sign Language Dataset",
)
MODEL_PATH = BASE_DIR / "sign_language_cnn.h5"
IMAGE_SHAPE = (28, 28, 1)


def find_dataset_dir() -> Path:
    for directory in DATA_DIR_CANDIDATES:
        if (
            (directory / "sign_mnist_train.csv").is_file()
            and (directory / "sign_mnist_test.csv").is_file()
        ):
            return directory

    searched = "\n".join(f"  - {directory}" for directory in DATA_DIR_CANDIDATES)
    raise FileNotFoundError(
        "Could not find both Sign Language MNIST CSV files. "
        f"Expected them in one of:\n{searched}"
    )


def load_dataset(data_dir: Path):
    train_df = pd.read_csv(data_dir / "sign_mnist_train.csv")
    test_df = pd.read_csv(data_dir / "sign_mnist_test.csv")

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Both CSV files must contain a 'label' column.")

    train_pixels = train_df.drop(columns="label")
    test_pixels = test_df.drop(columns="label")
    if train_pixels.shape[1] != 784 or test_pixels.shape[1] != 784:
        raise ValueError(
            "Expected exactly 784 pixel columns per CSV; "
            f"found {train_pixels.shape[1]} in training and "
            f"{test_pixels.shape[1]} in test data."
        )
    if not train_pixels.columns.equals(test_pixels.columns):
        raise ValueError("Training and test CSV pixel columns must match.")

    x_train = (
        train_pixels.to_numpy(dtype=np.float32, copy=True).reshape(-1, *IMAGE_SHAPE)
        / 255.0
    )
    x_test = (
        test_pixels.to_numpy(dtype=np.float32, copy=True).reshape(-1, *IMAGE_SHAPE)
        / 255.0
    )

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(train_df["label"])
    known_labels = set(label_binarizer.classes_)
    test_labels = set(test_df["label"].unique())
    unknown_labels = test_labels - known_labels
    if unknown_labels:
        raise ValueError(
            f"Test data contains labels absent from training data: "
            f"{sorted(unknown_labels)}"
        )
    y_test = label_binarizer.transform(test_df["label"])

    return x_train, y_train, x_test, y_test, label_binarizer.classes_


def build_model(num_classes: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=IMAGE_SHAPE),
            # Augment the 28x28 training images so the CNN becomes less sensitive
            # to small differences in position, scale, and left/right orientation.
            RandomRotation(0.08),
            RandomZoom(0.10),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    data_dir = find_dataset_dir()
    x_train, y_train, x_test, y_test, classes = load_dataset(data_dir)
    print(f"Loaded {len(x_train)} training and {len(x_test)} test images.")
    print(f"Labels represented: {classes.tolist()}")

    # Reserve a separate, stratified validation set before training.
    # It is kept out of training and used to stop or tune the model reliably.
    train_indices = np.argmax(y_train, axis=1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.10,
        random_state=42,
        stratify=train_indices,
    )
    print(
        f"Training split: {len(x_train)} images; "
        f"validation split: {len(x_val)} images."
    )

    model = build_model(num_classes=len(classes))
    history = model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=64,
        verbose=2,
        validation_data=(x_val, y_val),
        callbacks=[
            # Restore the model weights from the epoch with the best validation
            # accuracy instead of keeping the weights from the last epoch.
            EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
                mode="max",
            ),
            # Reduce the learning rate when validation loss stops improving.
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            ),
        ],
    )

    # Validation accuracy measures fitting quality during development.
    # Test accuracy is reported separately and must be measured only after training.
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Generate predictions for classification report and visualizations
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_true_idx = np.argmax(y_test, axis=1)

    # Map label indices (0-8, 10-24) to letters (A-I, K-Y)
    y_true_labels = [classes[i] for i in y_true_idx]
    y_pred_labels = [classes[i] for i in y_pred_idx]

    y_true_letters = [chr(ord("A") + lbl) for lbl in y_true_labels]
    y_pred_letters = [chr(ord("A") + lbl) for lbl in y_pred_labels]
    letter_labels = [chr(ord("A") + lbl) for lbl in classes]

    y_true_arr = np.array(y_true_letters)
    y_pred_arr = np.array(y_pred_letters)

    test_report = classification_report(
        y_true_letters,
        y_pred_letters,
        labels=letter_labels,
        target_names=letter_labels,
        output_dict=True,
        zero_division=0,
    )
    test_report_text = classification_report(
        y_true_letters,
        y_pred_letters,
        labels=letter_labels,
        target_names=letter_labels,
        zero_division=0,
    )
    macro_scores = test_report["macro avg"]
    weighted_scores = test_report["weighted avg"]
    metrics_report_text = (
        "TEST SET OVERALL METRICS\n"
        + "=" * 60
        + f"\nAccuracy: {test_accuracy:.4%}"
        + f"\nMacro precision: {macro_scores['precision']:.4f}"
        + f"\nMacro recall: {macro_scores['recall']:.4f}"
        + f"\nMacro F1-score: {macro_scores['f1-score']:.4f}"
        + f"\nWeighted precision: {weighted_scores['precision']:.4f}"
        + f"\nWeighted recall: {weighted_scores['recall']:.4f}"
        + f"\nWeighted F1-score: {weighted_scores['f1-score']:.4f}"
    )
    print("\nPer-class test metrics:")
    print(test_report_text)
    print("\n" + metrics_report_text)

    # 1. Training Curves (Accuracy & Loss side-by-side with Overfitting / Underfitting Diagnosis)
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    final_augmented_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    final_augmented_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    # Keras records training metrics while augmentation is active, but validation
    # metrics use original images. Evaluate the training split without augmentation
    # for a fair train-versus-validation comparison in the diagnostic.
    train_eval_loss, train_eval_acc = model.evaluate(x_train, y_train, verbose=0)
    acc_gap = train_eval_acc - final_val_acc

    # Diagnose overfitting from a widening train/validation gap or validation
    # performance declining; line connectivity itself is not evidence of it.
    validation_accuracy_drop = max(val_acc) - final_val_acc
    if train_eval_acc < 0.80 and final_val_acc < 0.80:
        fit_status = "Underfitting Detected"
        status_desc = "The model has not learned sufficient visual features (Accuracy < 80%)."
        status_color = "#c0392b"  # Red
    elif acc_gap > 0.08 or (
        validation_accuracy_drop > 0.05 and train_eval_acc > final_val_acc
    ):
        fit_status = "Overfitting Detected (Memorization)"
        status_desc = (
            f"Training/validation accuracy gap: {acc_gap * 100:.1f}%; "
            f"validation accuracy fell {validation_accuracy_drop * 100:.1f} "
            "points from its peak."
        )
        status_color = "#e67e22"  # Orange
    elif final_val_acc - test_accuracy > 0.05:
        fit_status = "Validation/Test Generalization Gap"
        status_desc = (
            f"Validation accuracy ({final_val_acc * 100:.1f}%) exceeds "
            f"independent test accuracy ({test_accuracy * 100:.1f}%) by "
            f"{(final_val_acc - test_accuracy) * 100:.1f} points."
        )
        status_color = "#e67e22"  # Orange
    else:
        fit_status = "No Clear Overfitting Signal"
        status_desc = (
            f"Unaugmented train/validation accuracy gap: "
            f"{acc_gap * 100:.1f}%; independent test accuracy: "
            f"{test_accuracy * 100:.1f}%."
        )
        status_color = "#1e8449"  # Green

    print("\n" + "=" * 70)
    print("TRAINING DIAGNOSTIC: OVERFITTING / UNDERFITTING ASSESSMENT")
    print("=" * 70)
    print(f"Final Augmented Training Accuracy: {final_augmented_train_acc * 100:.2f}%")
    print(f"Training Accuracy (no augmentation): {train_eval_acc * 100:.2f}%")
    print(f"Final Validation Accuracy  : {final_val_acc * 100:.2f}%")
    print(f"Generalization Accuracy Gap: {acc_gap * 100:.2f}%")
    print(f"Training Loss (no augmentation): {train_eval_loss:.4f}")
    print(f"Final Augmented Training Loss: {final_augmented_train_loss:.4f}")
    print(f"Final Validation Loss      : {final_val_loss:.4f}")
    print(f"Independent Test Accuracy  : {test_accuracy * 100:.2f}%")
    print(f"Diagnostic Status          : {fit_status}")
    print(f"Interpretation             : {status_desc}")
    print("=" * 70 + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Connect observed epoch measurements to show how each metric changes
    # throughout training; overfitting is assessed from metric divergence.
    ax1.plot(
        epochs_range,
        train_acc,
        "b-o",
        linewidth=2,
        markersize=6,
        label=f"Training Accuracy - augmented ({final_augmented_train_acc:.1%})",
    )
    ax1.plot(
        epochs_range,
        val_acc,
        "r--s",
        linewidth=2,
        markersize=6,
        label=f"Validation Accuracy ({final_val_acc:.1%})",
    )
    ax1.set_title("Training vs. Validation Accuracy", fontsize=13, weight="bold")
    ax1.set_xlabel("Epoch", fontsize=11, weight="bold")
    ax1.set_ylabel("Accuracy", fontsize=11, weight="bold")
    ax1.set_xticks(list(epochs_range))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=10)

    # Connect loss measurements for the same epoch-by-epoch view.
    ax2.plot(
        epochs_range,
        train_loss,
        "b-o",
        linewidth=2,
        markersize=6,
        label=f"Training Loss - augmented ({final_augmented_train_loss:.4f})",
    )
    ax2.plot(
        epochs_range,
        val_loss,
        "r--s",
        linewidth=2,
        markersize=6,
        label=f"Validation Loss ({final_val_loss:.4f})",
    )
    ax2.set_title("Training vs. Validation Loss", fontsize=13, weight="bold")
    ax2.set_xlabel("Epoch", fontsize=11, weight="bold")
    ax2.set_ylabel("Loss", fontsize=11, weight="bold")
    ax2.set_xticks(list(epochs_range))
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10)

    # Prominent header mentioning the Overfitting / Underfitting diagnosis
    fig.suptitle(
        f"Training Assessment: {fit_status}\n{status_desc}",
        fontsize=13,
        weight="bold",
        color=status_color,
        y=1.02,
    )

    plt.tight_layout()
    curves_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {curves_path}")

    # 2. Confusion Matrix Plot (Full 24x24)
    cm = confusion_matrix(y_true_letters, y_pred_letters, labels=letter_labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=letter_labels,
        yticklabels=letter_labels,
    )
    plt.title("Confusion Matrix (Sign Language MNIST)", fontsize=16, weight="bold", pad=12)
    plt.xlabel("Predicted Letter", fontsize=12, weight="bold")
    plt.ylabel("True Letter", fontsize=12, weight="bold")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # 3. Standalone 2x2 Matrix for Letter 'T' (One-vs-All)
    target_letter = "T"
    tp_t = int(np.sum((y_true_arr == target_letter) & (y_pred_arr == target_letter)))
    fn_t = int(np.sum((y_true_arr == target_letter) & (y_pred_arr != target_letter)))
    fp_t = int(np.sum((y_true_arr != target_letter) & (y_pred_arr == target_letter)))
    tn_t = int(np.sum((y_true_arr != target_letter) & (y_pred_arr != target_letter)))

    cm_2x2 = np.array([[tp_t, fn_t], [fp_t, tn_t]])
    annot_2x2 = [
        [f"TP\n{tp_t:,}", f"FN\n{fn_t:,}"],
        [f"FP\n{fp_t:,}", f"TN\n{tn_t:,}"],
    ]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_2x2,
        annot=np.array(annot_2x2),
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted: T", "Predicted: Not T"],
        yticklabels=["Actual: T", "Actual: Not T"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"One-vs-All Confusion Matrix: Letter '{target_letter}'", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12, weight="bold")
    plt.ylabel("Actual Class", fontsize=12, weight="bold")
    plt.tight_layout()
    cm_2x2_path = OUTPUT_DIR / "confusion_matrix_2x2.png"
    plt.savefig(cm_2x2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 2x2 confusion matrix to {cm_2x2_path}")

    # 4. Per-Class Confusion Matrix Metrics Table & CSV
    table_data = []
    for letter in letter_labels:
        is_true = y_true_arr == letter
        is_pred = y_pred_arr == letter
        c_tp = int(np.sum(is_true & is_pred))
        c_fn = int(np.sum(is_true & ~is_pred))
        c_fp = int(np.sum(~is_true & is_pred))
        c_tn = int(np.sum(~is_true & ~is_pred))
        actual = c_tp + c_fn
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / actual if actual > 0 else 0.0
        f1 = (
            2 * prec * rec / (prec + rec)
            if (prec + rec) > 0
            else 0.0
        )
        table_data.append(
            {
                "Letter": letter,
                "Actual (TP+FN)": actual,
                "TP": c_tp,
                "FP": c_fp,
                "FN": c_fn,
                "TN": c_tn,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-score": round(f1, 4),
            }
        )

    metrics_df = pd.DataFrame(table_data)
    csv_path = OUTPUT_DIR / "confusion_matrix_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics CSV to {csv_path}")

    # Render formatted table as confusion_matrix_table.png
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.axis("off")
    ax.axis("tight")

    display_rows = [
        [
            row["Letter"],
            f"{row['Actual (TP+FN)']:,}",
            f"{row['TP']:,}",
            f"{row['FP']:,}",
            f"{row['FN']:,}",
            f"{row['TN']:,}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1-score']:.4f}",
        ]
        for row in table_data
    ]
    table_columns = [
        "Letter",
        "Actual (TP+FN)",
        "TP",
        "FP",
        "FN",
        "TN",
        "Precision",
        "Recall",
        "F1-score",
    ]
    table = ax.table(
        cellText=display_rows,
        colLabels=table_columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.45)

    for (r_idx, c_idx), cell in table.get_celld().items():
        if r_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2c3e50")
        elif r_idx % 2 == 0:
            cell.set_facecolor("#f4f6f7")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#bdc3c7")

    plt.title(
        "Per-Class Confusion Matrix Metrics (ASL Alphabet)",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    plt.tight_layout()
    table_img_path = OUTPUT_DIR / "confusion_matrix_table.png"
    plt.savefig(table_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved table image to {table_img_path}")

    # 5. Representative Test Predictions (show errors as well as correct results)
    sample_count = min(25, len(x_test))
    mismatch_indices = np.flatnonzero(y_true_idx != y_pred_idx)
    correct_indices = np.flatnonzero(y_true_idx == y_pred_idx)
    rng = np.random.default_rng(42)

    # Put up to half the available slots toward actual test errors so a small
    # preview does not hide model mistakes just because early rows are easy.
    error_count = min((sample_count + 1) // 2, len(mismatch_indices))
    selected_errors = (
        rng.choice(mismatch_indices, size=error_count, replace=False)
        if error_count
        else np.array([], dtype=int)
    )
    correct_count = min(sample_count - error_count, len(correct_indices))
    selected_correct = (
        rng.choice(correct_indices, size=correct_count, replace=False)
        if correct_count
        else np.array([], dtype=int)
    )
    sample_indices = np.concatenate((selected_errors, selected_correct))
    rng.shuffle(sample_indices)

    plt.figure(figsize=(12, 12))
    for plot_index, test_index in enumerate(sample_indices):
        plt.subplot(5, 5, plot_index + 1)
        # Keep the original 28x28 pixels visible without interpolation blur.
        plt.imshow(
            x_test[test_index].reshape(28, 28),
            cmap="gray",
            interpolation="nearest",
        )
        true_char = y_true_letters[test_index]
        pred_char = y_pred_letters[test_index]
        color = "green" if true_char == pred_char else "red"
        confidence = float(np.max(y_pred_probs[test_index])) * 100.0
        plt.title(
            f"True: {true_char} | Pred: {pred_char}\n"
            f"Confidence: {confidence:.1f}%",
            color=color,
            fontsize=9,
            weight="bold",
        )
        plt.axis("off")
    plt.suptitle(
        "Representative Test Predictions (errors and correct)",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout()
    sample_path = OUTPUT_DIR / "sample_predictions.png"
    plt.savefig(sample_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sample predictions plot to {sample_path}")


if __name__ == "__main__":
    main()
