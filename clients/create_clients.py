import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    ROOT,
    "data",
    "raw",
    "UNSW_NB15_training-set.xlsx"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "data",
    "client_splits"
)

NUM_CLIENTS = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TRUSTFED-6G CLIENT DATA PARTITIONING")
print("=" * 70)

print("\nLoading:")
print(DATA_FILE)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

df = pd.read_excel(DATA_FILE)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "label"

if TARGET not in df.columns:
    raise ValueError(
        "Column 'label' was not found."
    )

y = df[TARGET].astype(int)

X = df.drop(
    columns=[TARGET]
)


# ============================================================
# REMOVE NON-PREDICTIVE / LEAKAGE COLUMNS
# ============================================================

if "id" in X.columns:
    X = X.drop(
        columns=["id"]
    )

if "attack_cat" in X.columns:
    X = X.drop(
        columns=["attack_cat"]
    )


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "category", "str"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumber of numerical columns:")
print(len(numerical_columns))


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# STRATIFIED GLOBAL SPLIT
# ============================================================

X_train, X_unused, y_train, y_unused = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("\nFederated training samples:")
print(len(X_train))

print("\nHeld-out samples:")
print(len(X_unused))


# ============================================================
# FIT PREPROCESSING ONLY ON TRAINING DATA
# ============================================================

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

y_train = np.asarray(
    y_train,
    dtype=np.int64
)

print(
    "\nProcessed training shape:",
    X_train_processed.shape
)


# ============================================================
# RANDOMIZED STRATIFIED CLIENT PARTITION
# ============================================================

print("\nCreating virtual IoT clients...")

indices = np.arange(
    len(X_train_processed)
)

# Shuffle deterministically
rng = np.random.default_rng(SEED)

rng.shuffle(indices)


# ============================================================
# SPLIT INDICES
# ============================================================

client_indices = np.array_split(
    indices,
    NUM_CLIENTS
)


# ============================================================
# CREATE CLIENT DATA FILES
# ============================================================

for client_id, idx in enumerate(
    client_indices
):

    client_X = X_train_processed[idx]

    client_y = y_train[idx]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"client_{client_id}.npz"
    )

    np.savez_compressed(
        output_file,
        X=client_X,
        y=client_y
    )

    print(
        f"Client {client_id:02d}: "
        f"{len(idx):6d} samples | "
        f"features={client_X.shape[1]}"
    )


# ============================================================
# SAVE GLOBAL TEST SET
# ============================================================

X_test = X_unused.copy()

y_test = np.asarray(
    y_unused,
    dtype=np.int64
)

X_test_processed = preprocessor.transform(
    X_test
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

test_file = os.path.join(
    OUTPUT_DIR,
    "global_test.npz"
)

np.savez_compressed(
    test_file,
    X=X_test_processed,
    y=y_test
)

print(
    "\nGlobal test set saved:",
    test_file
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

import joblib

preprocessor_file = os.path.join(
    OUTPUT_DIR,
    "preprocessor.joblib"
)

joblib.dump(
    preprocessor,
    preprocessor_file
)

print(
    "Preprocessor saved:",
    preprocessor_file
)


# ============================================================
# CREATE CLIENT METADATA
# ============================================================

metadata = []

for client_id, idx in enumerate(
    client_indices
):

    # Simulated heterogeneous IoT characteristics
    cpu_available = float(
        rng.uniform(0.40, 0.95)
    )

    memory_available = float(
        rng.uniform(0.40, 0.95)
    )

    bandwidth = float(
        rng.uniform(0.40, 0.95)
    )

    latency_score = float(
        rng.uniform(0.40, 0.95)
    )

    privacy_risk = float(
        rng.uniform(0.10, 0.90)
    )

    # Initial trust is neutral
    trust = 0.50

    metadata.append(
        {
            "client_id": client_id,
            "num_samples": len(idx),
            "trust": trust,
            "cpu_available": cpu_available,
            "memory_available": memory_available,
            "bandwidth": bandwidth,
            "latency_score": latency_score,
            "privacy_risk": privacy_risk
        }
    )


metadata_df = pd.DataFrame(
    metadata
)

metadata_file = os.path.join(
    OUTPUT_DIR,
    "client_metadata.csv"
)

metadata_df.to_csv(
    metadata_file,
    index=False
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "num_clients": NUM_CLIENTS,
    "seed": SEED,
    "training_samples": int(len(X_train_processed)),
    "test_samples": int(len(X_test_processed)),
    "feature_count": int(X_train_processed.shape[1]),
    "target": TARGET
}

config_file = os.path.join(
    OUTPUT_DIR,
    "federated_data_config.json"
)

import json

with open(
    config_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CLIENT DATA PARTITIONING COMPLETE")
print("=" * 70)

print("\nCreated:")
print(
    f"  {NUM_CLIENTS} virtual IoT client datasets"
)

print(
    "  1 global test dataset"
)

print(
    "  1 preprocessing pipeline"
)

print(
    "  client metadata"
)

print(
    "  federated configuration"
)

print("\nOutput directory:")
print(OUTPUT_DIR)