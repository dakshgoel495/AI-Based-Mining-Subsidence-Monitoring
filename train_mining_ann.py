"""Train the two ANN models from ann2.ipynb.

ANN 1: Normal (0) vs Abnormal (1)
ANN 2: Fault type classification (the original notebook includes Normal as class 7).

Example:
    python train_mining_ann.py --csv mining_sensor_fault_dataset.csv

Colab example:
    !python train_mining_ann.py --csv /content/mining_sensor_fault_dataset.csv

Requires: pandas, numpy, scikit-learn, tensorflow, matplotlib, joblib
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots without requiring a desktop window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

FAULT_MAPPING = {
    "Tilt_and_Distance": 0,
    "High_Vibration": 1,
    "Vibration_and_Distance": 2,
    "Multiple_Sensor_Abnormality": 3,
    "Vibration_and_Tilt": 4,
    "Excessive_Tilt": 5,
    "Abnormal_Ultrasonic_Distance": 6,
    "Normal": 7,
}
STATUS_MAPPING = {"Normal": 0, "Abnormal": 1}


def build_model(num_features, output_units, output_activation, loss):
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(output_units, activation=output_activation),
    ])
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def save_matrix(y_true, predictions, labels, names, title, path):
    cm = confusion_matrix(y_true, predictions, labels=labels)
    print(f"\n{title}\n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(xticks_rotation=90, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train mining status and fault ANNs")
    parser.add_argument("--csv", type=Path, default=Path("mining_sensor_fault_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("model_artifacts"))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Dataset not found: {args.csv}. Download/upload the CSV first.")
    args.output.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    tf.random.set_seed(42)
    df = pd.read_csv(args.csv)
    required = {"Node_ID", "Status", "Fault_Type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required CSV columns: {sorted(required - set(df.columns))}")

    # Same feature selection and labels as the original notebook.
    X = df.drop(columns=["Node_ID", "Status", "Fault_Type"])
    y_status = df["Status"].map(STATUS_MAPPING)
    y_fault = df["Fault_Type"].fillna("Normal").map(FAULT_MAPPING)
    if y_status.isna().any():
        raise ValueError("Unknown Status values found. Expected Normal or Abnormal.")
    if y_fault.isna().any():
        bad = df.loc[y_fault.isna(), "Fault_Type"].unique().tolist()
        raise ValueError(f"Unknown Fault_Type values: {bad}")
    if X.isna().any().any():
        raise ValueError("Feature data contains missing values; handle them before training.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All input feature columns must be numeric.")
    y_status = y_status.astype(int)
    y_fault = y_fault.astype(int)

    print("Rows, columns:", df.shape)
    print("Features:", X.columns.tolist())
    print("Status counts:\n", y_status.value_counts().sort_index())
    print("Fault counts:\n", y_fault.value_counts().sort_index())

    # Both models use the notebook's independent stratified 80/20 split.
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_status, test_size=0.2, random_state=42, stratify=y_status
    )
    scaler_s = StandardScaler()
    X_train_s = scaler_s.fit_transform(X_train_s)
    X_test_s = scaler_s.transform(X_test_s)

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X, y_fault, test_size=0.2, random_state=42, stratify=y_fault
    )
    scaler_f = StandardScaler()
    X_train_f = scaler_f.fit_transform(X_train_f)
    X_test_f = scaler_f.transform(X_test_f)

    print("\nTraining ANN 1: Normal / Abnormal")
    model_s = build_model(X_train_s.shape[1], 1, "sigmoid", "binary_crossentropy")
    model_s.fit(X_train_s, y_train_s, validation_split=0.2,
                epochs=args.epochs, batch_size=32, verbose=1)
    _, accuracy_s = model_s.evaluate(X_test_s, y_test_s, verbose=0)
    print(f"ANN 1 test accuracy: {accuracy_s:.2%}")
    pred_s = (model_s.predict(X_test_s, verbose=0).ravel() >= 0.5).astype(int)
    print(classification_report(y_test_s, pred_s, labels=[0, 1],
                                target_names=["Normal", "Abnormal"], zero_division=0))
    save_matrix(y_test_s, pred_s, [0, 1], ["Normal", "Abnormal"],
                "Status Model Confusion Matrix", args.output / "status_confusion_matrix.png")

    print("\nTraining ANN 2: Fault Type")
    # Preserve the original notebook's inclusion of Normal as class 7.
    # For a production pipeline, consider training ANN 2 on abnormal cases only.
    classes_present = sorted(y_fault.unique().tolist())
    if classes_present != list(range(len(classes_present))):
        raise ValueError("Fault class indices must be contiguous starting at 0.")
    model_f = build_model(X_train_f.shape[1], len(classes_present),
                          "softmax", "sparse_categorical_crossentropy")
    model_f.fit(X_train_f, y_train_f, validation_split=0.2,
                epochs=args.epochs, batch_size=32, verbose=1)
    _, accuracy_f = model_f.evaluate(X_test_f, y_test_f, verbose=0)
    print(f"ANN 2 test accuracy: {accuracy_f:.2%}")
    pred_f = model_f.predict(X_test_f, verbose=0).argmax(axis=1)
    fault_names = [next(name for name, number in FAULT_MAPPING.items() if number == i)
                   for i in classes_present]
    print(classification_report(y_test_f, pred_f, labels=classes_present,
                                target_names=fault_names, zero_division=0))
    save_matrix(y_test_f, pred_f, classes_present, fault_names,
                "Fault Model Confusion Matrix", args.output / "fault_confusion_matrix.png")

    # Save the same 4 backend model/preprocessing artifacts as the notebook.
    model_s.save(args.output / "status_model.keras")
    model_f.save(args.output / "fault_model.keras")
    joblib.dump(scaler_s, args.output / "scaler_s.pkl")
    joblib.dump(scaler_f, args.output / "scaler_f.pkl")
    with open(args.output / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"features": X.columns.tolist(), "status_mapping": STATUS_MAPPING,
                   "fault_mapping": FAULT_MAPPING, "test_accuracy_status": float(accuracy_s),
                   "test_accuracy_fault": float(accuracy_f),
                   "note": "Tested on a held-out split of the supplied dataset; not field-validated."}, f, indent=2)
    print(f"\nSaved all files in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
