import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# PATH SETUP
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    ROOT,
    "data",
    "raw",
    "UNSW_NB15_training-set.xlsx"
)

RESULTS_DIR = os.path.join(ROOT, "results", "metrics")
MODEL_DIR = os.path.join(ROOT, "results", "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("TRUSTFED-6G CENTRALIZED BASELINE")
    print("=" * 70)

    print("\nLoading dataset:")
    print(DATA_FILE)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}\n"
        )

    df = pd.read_excel(DATA_FILE)

    print("\nDataset shape:", df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    return df


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_dataset(df):

    print("\n" + "=" * 70)
    print("PREPROCESSING")
    print("=" * 70)

    target_column = "label"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    # --------------------------------------------------------
    # Remove target and non-feature attack category
    # --------------------------------------------------------

    y = df[target_column].astype(int)

    X = df.drop(columns=[target_column])

    # attack_cat is descriptive attack-class information.
    # We exclude it from binary intrusion prediction to avoid
    # target leakage.
    if "attack_cat" in X.columns:
        X = X.drop(columns=["attack_cat"])

    # ID is an identifier, not a meaningful predictive feature.
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    print("\nTarget distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # Identify categorical and numerical features
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumber of numerical columns:")
    print(len(numerical_columns))

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # Combined preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            ),
        ]
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # --------------------------------------------------------
    # Fit preprocessing only on training data
    # --------------------------------------------------------

    print("\nFitting preprocessing pipeline...")

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int64
    )

    y_test = np.asarray(
        y_test,
        dtype=np.int64
    )

    print(
        "\nProcessed training shape:",
        X_train_processed.shape
    )

    print(
        "Processed testing shape:",
        X_test_processed.shape
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# MLP MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch).squeeze(1)

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * len(y_batch)
            )

            predictions = (
                torch.sigmoid(outputs) >= 0.5
            ).long()

            correct += (
                predictions == y_batch.long()
            ).sum().item()

            total += len(y_batch)

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy:.4f}"
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model(X_tensor)

        probabilities = torch.sigmoid(
            outputs.squeeze(1)
        )

        predictions = (
            probabilities >= 0.5
        ).cpu().numpy().astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    metrics = {
        "model": "Centralized MLP",
        "dataset": "UNSW-NB15",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    }

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(model, preprocessor, metrics):

    metrics_file = os.path.join(
        RESULTS_DIR,
        "centralized_baseline.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    model_file = os.path.join(
        MODEL_DIR,
        "centralized_mlp.pth"
    )

    torch.save(
        model.state_dict(),
        model_file
    )

    print("\nResults saved to:")
    print(metrics_file)

    print("\nModel saved to:")
    print(model_file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nUsing device:", DEVICE)

    df = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_dataset(df)

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    input_size = X_train.shape[1]

    print("\nMLP input size:", input_size)

    model = MLP(
        input_size=input_size
    ).to(DEVICE)

    train_model(
        model,
        train_loader
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_results(
        model,
        preprocessor,
        metrics
    )

    print("\n" + "=" * 70)
    print("CENTRALIZED BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()