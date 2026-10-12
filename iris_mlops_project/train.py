from pathlib import Path
import json
import joblib
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "iris_dataset.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
FEATURES = ["sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)"]
TARGET = "species"

def main():
    df = pd.read_csv(DATA)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing}. Found: {list(df.columns)}")
    df = df[FEATURES + [TARGET]].dropna()
    X = df[FEATURES].astype("float32").to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
    ])
    lr, epochs, batch_size = 0.001, 100, 16
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    mlflow.set_experiment("Iris-Neural-Network-MLOps")
    with mlflow.start_run():
        mlflow.log_params({
            "dataset": DATA.name, "test_size": 0.2, "random_state": 10,
            "hidden_layer_1_neurons": 32, "hidden_layer_2_neurons": 16,
            "activation": "relu", "output_activation": "softmax",
            "optimizer": "Adam", "learning_rate": lr, "epochs": epochs,
            "batch_size": batch_size, "scaling": "StandardScaler"
        })
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  verbose=0, validation_split=0.1)
        pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = float(accuracy_score(y_test, pred))
        report = classification_report(y_test, pred, target_names=encoder.classes_,
                                       output_dict=True, zero_division=0)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_samples", len(y_test))
        mlflow.log_metric("correct_predictions", int((pred == y_test).sum()))

        model_path = OUT / "iris_nn.keras"
        scaler_path = OUT / "scaler.joblib"
        encoder_path = OUT / "label_encoder.joblib"
        metrics_path = OUT / "metrics.json"
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder, encoder_path)
        metrics_path.write_text(json.dumps({
            "test_accuracy_percent": round(acc * 100, 2),
            "test_samples": int(len(y_test)),
            "correct_predictions": int((pred == y_test).sum()),
            "classes": encoder.classes_.tolist(),
            "classification_report": report
        }, indent=2), encoding="utf-8")
        for path in [model_path, scaler_path, encoder_path]:
            mlflow.log_artifact(str(path), artifact_path="model_files")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        print(f"Dataset rows: {len(df)}")
        print(f"Training samples: {len(y_train)} | Test samples: {len(y_test)}")
        print(f"Test accuracy: {acc * 100:.2f}%")
        print(f"Model saved in: {model_path}")
        print("MLflow logged this experiment. Run `mlflow ui` to view it.")

if __name__ == "__main__":
    main()
