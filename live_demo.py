from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "live_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "sign_language_cnn.h5"
TASK_PATH = BASE_DIR / "hand_landmarker.task"

# Sign Language MNIST labels (0-8 -> A-I, 10-24 -> K-Y; J and Z require motion)
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
LABELS = [chr(ord("A") + c) for c in CLASSES]
HISTORY_FRAMES = 8
MIN_HISTORY_FRAMES = 3
MIN_AGREEMENT = 0.60
MIN_CONFIDENCE = 55.0
MIN_CONFIDENCE_MARGIN = 5.0
STABLE_SECONDS = 0.75


def preprocess_hand_crop(frame: np.ndarray, hand_landmarks, padding: int = 28):
    """Crops a square region around the hand landmarks, matching the Sign Language MNIST format."""
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    box_half = max(max_x - min_x, max_y - min_y) // 2 + padding

    x1 = max(0, center_x - box_half)
    y1 = max(0, center_y - box_half)
    x2 = min(w, center_x + box_half)
    y2 = min(h, center_y + box_half)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    # Standard grayscale and resize to 28x28 (matches training dataset preprocessing)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32).reshape(1, 28, 28, 1) / 255.0
    return normalized, (x1, y1, x2, y2)


def predict_hand(model, inp_normalized):
    """Predicts one crop in the same orientation used by the training data."""
    probabilities = model.predict(inp_normalized, verbose=0)[0]
    ranked = np.argsort(probabilities)[::-1]
    idx = int(ranked[0])
    best_conf = float(probabilities[idx]) * 100.0
    runner_up_conf = float(probabilities[ranked[1]]) * 100.0
    confidence_margin = best_conf - runner_up_conf
    letter = LABELS[idx] if idx < len(LABELS) else "?"
    return letter, best_conf, confidence_margin, inp_normalized


def save_session_outputs(log_records: list) -> None:
    """Exports CSV log, text report, and summary plot to live_outputs/."""
    if not log_records:
        print("No confirmed sign detections logged during this session.")
        return

    df = pd.DataFrame(log_records)
    csv_path = OUTPUT_DIR / "live_predictions_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved live predictions log to {csv_path}")

    report_path = OUTPUT_DIR / "live_evaluation_report.txt"
    letter_counts = df["letter"].value_counts()
    avg_conf = df["confidence"].mean()
    total_detections = len(df)

    report_lines = [
        "=" * 60,
        "LIVE SIGN LANGUAGE DEMO - SESSION SUMMARY",
        "=" * 60,
        "",
        f"Session Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Confirmed Sign Events: {total_detections}",
        f"Average Verified Confidence: {avg_conf:.2f}%",
        "",
        "Detected Signs Breakdown:",
    ]
    for letter, count in letter_counts.items():
        pct = (count / total_detections) * 100.0
        report_lines.append(
            f"  Letter '{letter}': {count} event(s) ({pct:.1f}%)"
        )
    report_lines.extend(
        [
            "",
            "Accuracy, precision, recall, and F1-score are not calculated for "
            "live detections because the session does not collect ground-truth "
            "labels for the signs shown to the camera.",
            "=" * 60,
        ]
    )
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Saved live evaluation report to {report_path}")
    print("\n" + report_text)

    summary_chart_path = OUTPUT_DIR / "live_detection_summary.png"
    plt.figure(figsize=(10, 5))
    letter_counts.plot(kind="bar", color="#2c3e50", edgecolor="#34495e")
    plt.title("Live Demo - Verified Detected Signs Frequency", fontsize=14, weight="bold")
    plt.xlabel("ASL Letter", fontsize=12, weight="bold")
    plt.ylabel("Detection Count", fontsize=12, weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(summary_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved live detection summary chart to {summary_chart_path}")


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not TASK_PATH.is_file():
        raise FileNotFoundError(f"Missing MediaPipe task file: {TASK_PATH}")

    print("Loading CNN model...")
    model = load_model(MODEL_PATH)
    print("CNN model loaded successfully.")

    base_options = python.BaseOptions(model_asset_path=str(TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    timestamp_ms = 0
    history_buffer = deque(maxlen=HISTORY_FRAMES)
    log_records = []
    status_banner = ""
    status_timer = 0.0
    stable_letter = None
    stable_since = 0.0
    last_logged_letter = None

    print(f"\nLive demo running. All captured outputs will be saved to: {OUTPUT_DIR}")
    print("Controls: SPACE = Capture snapshot to live_outputs | Q = Quit & Save report\n")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Keep the camera orientation unchanged; mirroring can change the
            # ASL class and the CNN was trained on unmirrored Sign-MNIST images.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_letter = None
            current_conf = 0.0
            current_inp = None
            box_color = (0, 0, 255)  # Red default (no hand)
            display_text = "No hand detected"

            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                inp, box = preprocess_hand_crop(frame, hand_lms)

                if inp is not None and box is not None:
                    x1, y1, x2, y2 = box

                    # Predict from the unmirrored crop, then stabilize across frames.
                    raw_letter, raw_conf, confidence_margin, best_inp = predict_hand(model, inp)
                    current_inp = best_inp

                    # Add prediction to rolling temporal history buffer
                    history_buffer.append((raw_letter, raw_conf, confidence_margin))

                    # Multi-frame consensus recheck (comparing last 8 frames)
                    letters_in_window = [item[0] for item in history_buffer]
                    counts = Counter(letters_in_window)
                    most_common_letter, top_count = counts.most_common(1)[0]
                    agreement_ratio = top_count / len(history_buffer)
                    letter_confs = [c for l, c, margin in history_buffer if l == most_common_letter]
                    letter_margins = [margin for l, c, margin in history_buffer if l == most_common_letter]
                    avg_conf = float(np.mean(letter_confs))
                    avg_margin = float(np.mean(letter_margins))

                    # Confirm after a short, consistent consensus; display the
                    # measured gates during rechecking so a stuck state is diagnosable.
                    if (
                        len(history_buffer) >= MIN_HISTORY_FRAMES
                        and agreement_ratio >= MIN_AGREEMENT
                        and avg_conf >= MIN_CONFIDENCE
                        and avg_margin >= MIN_CONFIDENCE_MARGIN
                    ):
                        current_letter = most_common_letter
                        current_conf = avg_conf

                        # Require a brief stable hold to avoid logging transient
                        # frame predictions as separate sign events.
                        now = time.time()
                        if stable_letter != most_common_letter:
                            stable_letter = most_common_letter
                            stable_since = now
                            last_logged_letter = None

                        if now - stable_since >= STABLE_SECONDS:
                            box_color = (0, 255, 0)  # Green = Verified & Stable
                            display_text = f"Sign: {current_letter} ({current_conf:.1f}%) [Verified]"

                        if now - stable_since >= STABLE_SECONDS and last_logged_letter != current_letter:
                            log_records.append(
                                {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                                    "letter": current_letter,
                                    "confidence": round(current_conf, 2),
                                }
                            )
                            last_logged_letter = current_letter
                        elif now - stable_since < STABLE_SECONDS:
                            box_color = (0, 215, 255)
                            display_text = f"Sign: {most_common_letter} (Holding...)"
                    else:
                        current_letter = most_common_letter
                        current_conf = avg_conf
                        box_color = (0, 215, 255)  # Amber/Yellow = Rechecking & Stabilizing
                        display_text = (
                            f"Rechecking {most_common_letter}: "
                            f"agree {agreement_ratio:.0%}, conf {avg_conf:.0f}%, "
                            f"margin {avg_margin:.0f}%"
                        )
                        stable_letter = None
                        stable_since = 0.0
                        last_logged_letter = None

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            else:
                history_buffer.clear()
                stable_letter = None
                stable_since = 0.0
                last_logged_letter = None

            # Top instructions banner
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (30, 30, 30), -1)
            cv2.putText(
                frame,
                "SPACE: Capture Snapshot to live_outputs | Q: Quit & Save Report",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            # Display prediction banner with matching state color
            cv2.putText(
                frame,
                display_text,
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                box_color,
                2,
                cv2.LINE_AA,
            )

            # Temporary capture confirmation notification
            if time.time() < status_timer and status_banner:
                cv2.putText(
                    frame,
                    status_banner,
                    (20, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Sign Language Recognizer - Live Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord(" ") and current_letter is not None:
                # Capture snapshot and crisp crop to live_outputs
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = OUTPUT_DIR / f"capture_{current_letter}_{ts_str}.png"
                cv2.imwrite(str(snap_path), frame)

                if current_inp is not None:
                    crop_28x28 = (current_inp[0, :, :, 0] * 255.0).astype(np.uint8)
                    crop_hires = cv2.resize(crop_28x28, (280, 280), interpolation=cv2.INTER_NEAREST)
                    crop_path = OUTPUT_DIR / f"crop_{current_letter}_{ts_str}.png"
                    cv2.imwrite(str(crop_path), crop_hires)

                status_banner = f"Captured '{current_letter}' ({current_conf:.1f}%) to live_outputs/"
                status_timer = time.time() + 2.0
                print(f"Captured snapshot: {snap_path.name}")

    cap.release()
    cv2.destroyAllWindows()

    # Save summary report, CSV log, and charts to live_outputs
    save_session_outputs(log_records)


if __name__ == "__main__":
    main()
