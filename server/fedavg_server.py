import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# TRUSTFED-6G
# FEDAVG BASELINE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CLIENT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "client_splits"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

METRICS_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

MODELS_DIR = os.path.join(
    RESULTS_DIR,
    "models"
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 10
ROUNDS = 5

LOCAL_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUSTFED-6G FEDAVG BASELINE")
print("=" * 70)

print(f"Clients       : {NUM_CLIENTS}")
print(f"Rounds        : {ROUNDS}")
print(f"Local epochs  : {LOCAL_EPOCHS}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Learning rate : {LEARNING_RATE}")
print(f"Device        : {DEVICE}")
print()


# ============================================================
# MLP MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_dim=192):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# LOAD CLIENT DATA
# ============================================================

def load_client(client_id):

    path = os.path.join(
        CLIENT_DIR,
        f"client_{client_id}.npz"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Client dataset not found: {path}"
        )

    data = np.load(path)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    ).reshape(-1, 1)

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    return dataset


# ============================================================
# GET MODEL PARAMETERS
# ============================================================

def get_parameters(model):

    return [
        value.detach().cpu().numpy().copy()
        for value in model.state_dict().values()
    ]


# ============================================================
# SET MODEL PARAMETERS
# ============================================================

def set_parameters(model, parameters):

    state_dict = model.state_dict()

    for key, value in zip(
        state_dict.keys(),
        parameters
    ):

        state_dict[key] = torch.tensor(
            value,
            dtype=state_dict[key].dtype
        )

    model.load_state_dict(
        state_dict
    )


# ============================================================
# LOCAL CLIENT TRAINING
# ============================================================

def train_client(
    client_id,
    global_parameters
):

    dataset = load_client(
        client_id
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = MLP(
        input_dim=192
    )

    set_parameters(
        model,
        global_parameters
    )

    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    total_loss = 0.0

    for epoch in range(
        LOCAL_EPOCHS
    ):

        epoch_loss = 0.0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            output = model(
                X_batch
            )

            loss = criterion(
                output,
                y_batch
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        total_loss = (
            epoch_loss /
            len(loader)
        )

    print(
        f"  Client {client_id:02d} | "
        f"samples={len(dataset):5d} | "
        f"loss={total_loss:.4f}"
    )

    return (
        get_parameters(model),
        len(dataset)
    )


# ============================================================
# FEDAVG AGGREGATION
# ============================================================

def fedavg(
    client_parameters,
    client_sizes
):

    total_samples = sum(
        client_sizes
    )

    averaged_parameters = []

    for parameter_index in range(
        len(client_parameters[0])
    ):

        weighted_sum = None

        for client_index in range(
            len(client_parameters)
        ):

            parameter = (
                client_parameters[
                    client_index
                ][parameter_index]
            )

            weight = (
                client_sizes[client_index]
                / total_samples
            )

            contribution = (
                parameter * weight
            )

            if weighted_sum is None:

                weighted_sum = contribution

            else:

                weighted_sum += contribution

        averaged_parameters.append(
            weighted_sum
        )

    return averaged_parameters


# ============================================================
# LOAD GLOBAL TEST DATA
# ============================================================

def load_global_test():

    path = os.path.join(
        CLIENT_DIR,
        "global_test.npz"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Global test dataset not found: {path}"
        )

    data = np.load(path)

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.float32
    )

    return X, y


# ============================================================
# GLOBAL MODEL EVALUATION
# ============================================================

def evaluate(
    model,
    X,
    y
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = model(
            X_tensor
        )

        probabilities = torch.sigmoid(
            outputs
        ).numpy().reshape(-1)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    return {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "confusion_matrix":
            cm.tolist()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD GLOBAL TEST SET
    # --------------------------------------------------------

    print(
        "Loading global test dataset..."
    )

    X_test, y_test = load_global_test()

    print(
        f"Global test samples: "
        f"{len(X_test)}"
    )

    print(
        f"Feature dimension: "
        f"{X_test.shape[1]}"
    )

    print()

    # --------------------------------------------------------
    # CREATE GLOBAL MODEL
    # --------------------------------------------------------

    global_model = MLP(
        input_dim=X_test.shape[1]
    )

    global_parameters = get_parameters(
        global_model
    )

    round_results = []

    # ========================================================
    # FEDERATED TRAINING
    # ========================================================

    for round_number in range(
        1,
        ROUNDS + 1
    ):

        print("=" * 70)

        print(
            f"FEDERATED ROUND "
            f"{round_number}/{ROUNDS}"
        )

        print("=" * 70)

        client_parameters = []

        client_sizes = []

        # ----------------------------------------------------
        # LOCAL CLIENT TRAINING
        # ----------------------------------------------------

        for client_id in range(
            NUM_CLIENTS
        ):

            parameters, size = train_client(
                client_id,
                global_parameters
            )

            client_parameters.append(
                parameters
            )

            client_sizes.append(
                size
            )

        # ----------------------------------------------------
        # FEDAVG
        # ----------------------------------------------------

        print()

        print(
            "Aggregating client models "
            "using FedAvg..."
        )

        global_parameters = fedavg(
            client_parameters,
            client_sizes
        )

        # ----------------------------------------------------
        # UPDATE GLOBAL MODEL
        # ----------------------------------------------------

        set_parameters(
            global_model,
            global_parameters
        )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        metrics = evaluate(
            global_model,
            X_test,
            y_test
        )

        print()

        print(
            f"Round {round_number} Results:"
        )

        print(
            f"Accuracy  : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{metrics['f1']:.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            np.array(
                metrics[
                    "confusion_matrix"
                ]
            )
        )

        round_results.append({

            "round":
                round_number,

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "confusion_matrix":
                metrics["confusion_matrix"]

        })

        print()

    # ========================================================
    # SAVE GLOBAL MODEL
    # ========================================================

    model_path = os.path.join(
        MODELS_DIR,
        "fedavg_global_model.pth"
    )

    torch.save(
        global_model.state_dict(),
        model_path
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = os.path.join(
        METRICS_DIR,
        "fedavg_baseline.json"
    )

    results = {

        "experiment":
            "FedAvg Baseline",

        "dataset":
            "UNSW-NB15",

        "num_clients":
            NUM_CLIENTS,

        "rounds":
            ROUNDS,

        "local_epochs":
            LOCAL_EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "feature_dimension":
            int(X_test.shape[1]),

        "test_samples":
            int(len(X_test)),

        "round_results":
            round_results,

        "final_results":
            round_results[-1]

    }

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("=" * 70)
    print("FEDAVG BASELINE COMPLETE")
    print("=" * 70)

    print()

    print(
        "Model saved:"
    )

    print(
        model_path
    )

    print()

    print(
        "Metrics saved:"
    )

    print(
        metrics_path
    )

    print()

    final = round_results[-1]

    print(
        "Final Results:"
    )

    print(
        f"Accuracy  : "
        f"{final['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{final['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{final['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{final['f1']:.4f}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()