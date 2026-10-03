import os
import json
import numpy as np
import pandas as pd
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

from trust.tprs import calculate_tprs


# ============================================================
# TRUSTFED-6G
# TPRS-BASED FEDERATED LEARNING
# ============================================================


# ============================================================
# PROJECT PATHS
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

os.makedirs(
    METRICS_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 10

ROUNDS = 5

CLIENTS_PER_ROUND = 7

LOCAL_EPOCHS = 1

BATCH_SIZE = 256

LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


# ============================================================
# TPRS WEIGHTS
# ============================================================

TRUST_WEIGHT = 0.30

QUALITY_WEIGHT = 0.25

RESOURCE_WEIGHT = 0.20

NETWORK_WEIGHT = 0.15

PRIVACY_WEIGHT = 0.10


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "TRUSTFED-6G TPRS FEDERATED LEARNING"
)

print("=" * 70)

print(
    f"Total clients       : {NUM_CLIENTS}"
)

print(
    f"Selected clients    : {CLIENTS_PER_ROUND}"
)

print(
    f"Rounds              : {ROUNDS}"
)

print(
    f"Local epochs        : {LOCAL_EPOCHS}"
)

print(
    f"Batch size          : {BATCH_SIZE}"
)

print(
    f"Learning rate       : {LEARNING_RATE}"
)

print(
    f"Device              : {DEVICE}"
)

print()


# ============================================================
# MLP MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim=192
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


# ============================================================
# LOAD CLIENT DATA
# ============================================================

def load_client(
    client_id
):

    path = os.path.join(
        CLIENT_DIR,
        f"client_{client_id}.npz"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Client dataset not found: {path}"
        )

    data = np.load(path)

    X = data[
        "X"
    ].astype(
        np.float32
    )

    y = data[
        "y"
    ].astype(
        np.float32
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    ).reshape(
        -1,
        1
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    return dataset


# ============================================================
# GET MODEL PARAMETERS
# ============================================================

def get_parameters(
    model
):

    return [

        value.detach()
        .cpu()
        .numpy()
        .copy()

        for value
        in model.state_dict().values()

    ]


# ============================================================
# SET MODEL PARAMETERS
# ============================================================

def set_parameters(
    model,
    parameters
):

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

    model.to(
        DEVICE
    )

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

    return (
        get_parameters(model),
        len(dataset),
        total_loss
    )


# ============================================================
# LOAD CLIENT METADATA
# ============================================================

def load_metadata():

    path = os.path.join(
        CLIENT_DIR,
        "client_metadata.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    metadata = pd.read_csv(
        path
    )

    required_columns = [

        "client_id",

        "num_samples",

        "trust",

        "cpu_available",

        "memory_available",

        "bandwidth",

        "latency_score",

        "privacy_risk"

    ]

    missing_columns = [

        column

        for column
        in required_columns

        if column not in metadata.columns

    ]

    if missing_columns:

        raise ValueError(
            "Missing metadata columns: "
            + str(missing_columns)
        )

    return metadata


# ============================================================
# RESOURCE SCORE
# ============================================================

def calculate_resource_score(
    cpu_available,
    memory_available
):

    cpu = float(
        cpu_available
    )

    memory = float(
        memory_available
    )

    score = (
        0.5 * cpu
        +
        0.5 * memory
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# NETWORK SCORE
# ============================================================

def calculate_network_score(
    bandwidth,
    latency_score
):

    bandwidth = float(
        bandwidth
    )

    latency = float(
        latency_score
    )

    score = (
        0.5 * bandwidth
        +
        0.5 * latency
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    loss
):

    loss = float(
        loss
    )

    # Lower loss means better local quality.
    score = 1.0 / (
        1.0 + loss
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# TPRS AGGREGATION
# ============================================================

def tprs_aggregate(
    client_parameters,
    client_sizes,
    tprs
):

    total_weight = sum(

        tprs[i]
        *
        client_sizes[i]

        for i
        in range(
            len(client_parameters)
        )

    )

    if total_weight <= 0:

        raise ValueError(
            "Total TPRS aggregation weight is zero."
        )

    aggregated_parameters = []

    for parameter_index in range(

        len(
            client_parameters[0]
        )

    ):

        aggregated = None

        for client_index in range(

            len(
                client_parameters
            )

        ):

            parameter = (

                client_parameters[
                    client_index
                ][
                    parameter_index
                ]

            )

            weight = (

                tprs[
                    client_index
                ]

                *

                client_sizes[
                    client_index
                ]

                /

                total_weight

            )

            contribution = (
                parameter * weight
            )

            if aggregated is None:

                aggregated = contribution

            else:

                aggregated += contribution

        aggregated_parameters.append(
            aggregated
        )

    return aggregated_parameters


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
            f"Global test file not found: {path}"
        )

    data = np.load(
        path
    )

    X = data[
        "X"
    ].astype(
        np.float32
    )

    y = data[
        "y"
    ].astype(
        np.float32
    )

    return X, y


# ============================================================
# EVALUATE GLOBAL MODEL
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

        output = model(
            X_tensor
        )

        probabilities = torch.sigmoid(
            output
        ).cpu().numpy().reshape(
            -1
        )

    predictions = (
        probabilities >= 0.5
    ).astype(
        int
    )

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

        "accuracy":
            float(
                accuracy
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "confusion_matrix":
            cm.tolist()

    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD METADATA
    # ========================================================

    print(
        "Loading client metadata..."
    )

    metadata = load_metadata()

    print(
        f"Metadata records: "
        f"{len(metadata)}"
    )

    print()

    print(
        "Client information:"
    )

    print(
        metadata.to_string(
            index=False
        )
    )

    print()

    # ========================================================
    # LOAD GLOBAL TEST DATA
    # ========================================================

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

    # ========================================================
    # INITIAL GLOBAL MODEL
    # ========================================================

    global_model = MLP(
        input_dim=X_test.shape[1]
    )

    global_parameters = get_parameters(
        global_model
    )

    round_results = []

    # ========================================================
    # FEDERATED ROUNDS
    # ========================================================

    for round_number in range(
        1,
        ROUNDS + 1
    ):

        print("=" * 70)

        print(
            f"TRUSTFED ROUND "
            f"{round_number}/{ROUNDS}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # LOCAL TRAINING
        # ----------------------------------------------------

        local_results = []

        for client_id in range(
            NUM_CLIENTS
        ):

            parameters, size, loss = train_client(

                client_id,

                global_parameters

            )

            local_results.append({

                "client_id":
                    client_id,

                "parameters":
                    parameters,

                "size":
                    size,

                "loss":
                    loss

            })

        # ----------------------------------------------------
        # CALCULATE CLIENT SCORES
        # ----------------------------------------------------

        client_scores = []

        for result in local_results:

            client_id = result[
                "client_id"
            ]

            rows = metadata[
                metadata[
                    "client_id"
                ]
                ==
                client_id
            ]

            if len(rows) == 0:

                raise ValueError(
                    f"No metadata for "
                    f"client {client_id}"
                )

            row = rows.iloc[0]

            # ------------------------------------------------
            # TRUST
            # ------------------------------------------------

            trust = float(
                row["trust"]
            )

            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            quality = (
                calculate_quality_score(
                    result["loss"]
                )
            )

            # ------------------------------------------------
            # RESOURCE
            # ------------------------------------------------

            resource = (
                calculate_resource_score(

                    row[
                        "cpu_available"
                    ],

                    row[
                        "memory_available"
                    ]

                )
            )

            # ------------------------------------------------
            # NETWORK
            # ------------------------------------------------

            network = (
                calculate_network_score(

                    row[
                        "bandwidth"
                    ],

                    row[
                        "latency_score"
                    ]

                )
            )

            # ------------------------------------------------
            # PRIVACY RISK
            # ------------------------------------------------

            privacy_risk = float(
                row[
                    "privacy_risk"
                ]
            )

            privacy_suitability = (
                1.0 - privacy_risk
            )

            # ------------------------------------------------
            # TPRS
            # ------------------------------------------------

            tprs = calculate_tprs(

                trust=trust,

                quality=quality,

                resource=resource,

                network=network,

                privacy_suitability=
                    privacy_suitability

            )

            client_scores.append({

                "client_id":
                    client_id,

                "trust":
                    trust,

                "quality":
                    quality,

                "resource":
                    resource,

                "network":
                    network,

                "privacy_risk":
                    privacy_risk,

                "tprs":
                    tprs,

                "parameters":
                    result[
                        "parameters"
                    ],

                "size":
                    result[
                        "size"
                    ],

                "loss":
                    result[
                        "loss"
                    ]

            })

        # ----------------------------------------------------
        # RANK CLIENTS
        # ----------------------------------------------------

        client_scores.sort(

            key=lambda item:
                item["tprs"],

            reverse=True

        )

        print()

        print(
            "TPRS CLIENT RANKING:"
        )

        print(
            "-" * 80
        )

        print(
            "Client | Trust | Quality | "
            "Resource | Network | Privacy | TPRS"
        )

        print(
            "-" * 80
        )

        for client in client_scores:

            print(

                f"{client['client_id']:6d} | "

                f"{client['trust']:.3f} | "

                f"{client['quality']:.3f} | "

                f"{client['resource']:.3f} | "

                f"{client['network']:.3f} | "

                f"{client['privacy_risk']:.3f} | "

                f"{client['tprs']:.3f}"

            )

        # ----------------------------------------------------
        # SELECT TOP CLIENTS
        # ----------------------------------------------------

        selected_clients = client_scores[
            :CLIENTS_PER_ROUND
        ]

        selected_ids = [

            client[
                "client_id"
            ]

            for client
            in selected_clients

        ]

        print()

        print(
            "Selected clients:"
        )

        print(
            selected_ids
        )

        # ----------------------------------------------------
        # AGGREGATION INPUTS
        # ----------------------------------------------------

        selected_parameters = [

            client[
                "parameters"
            ]

            for client
            in selected_clients

        ]

        selected_sizes = [

            client[
                "size"
            ]

            for client
            in selected_clients

        ]

        selected_tprs = [

            client[
                "tprs"
            ]

            for client
            in selected_clients

        ]

        # ----------------------------------------------------
        # TPRS-WEIGHTED AGGREGATION
        # ----------------------------------------------------

        print()

        print(
            "Performing TPRS-weighted aggregation..."
        )

        global_parameters = (
            tprs_aggregate(

                selected_parameters,

                selected_sizes,

                selected_tprs

            )
        )

        # ----------------------------------------------------
        # UPDATE GLOBAL MODEL
        # ----------------------------------------------------

        set_parameters(

            global_model,

            global_parameters

        )

        # ----------------------------------------------------
        # GLOBAL EVALUATION
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

        # ----------------------------------------------------
        # SAVE ROUND RESULTS
        # ----------------------------------------------------

        round_results.append({

            "round":
                round_number,

            "selected_clients":
                selected_ids,

            "client_scores": [

                {

                    "client_id":
                        client[
                            "client_id"
                        ],

                    "trust":
                        client[
                            "trust"
                        ],

                    "quality":
                        client[
                            "quality"
                        ],

                    "resource":
                        client[
                            "resource"
                        ],

                    "network":
                        client[
                            "network"
                        ],

                    "privacy_risk":
                        client[
                            "privacy_risk"
                        ],

                    "tprs":
                        client[
                            "tprs"
                        ]

                }

                for client
                in client_scores

            ],

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "f1":
                metrics[
                    "f1"
                ],

            "confusion_matrix":
                metrics[
                    "confusion_matrix"
                ]

        })

        print()

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(

        MODELS_DIR,

        "trustfed_tprs_model.pth"

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

        "trustfed_tprs.json"

    )

    results = {

        "experiment":
            "TrustFed-6G TPRS",

        "dataset":
            "UNSW-NB15",

        "num_clients":
            NUM_CLIENTS,

        "selected_clients_per_round":
            CLIENTS_PER_ROUND,

        "rounds":
            ROUNDS,

        "local_epochs":
            LOCAL_EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "tprs_weights": {

            "trust":
                TRUST_WEIGHT,

            "quality":
                QUALITY_WEIGHT,

            "resource":
                RESOURCE_WEIGHT,

            "network":
                NETWORK_WEIGHT,

            "privacy":
                PRIVACY_WEIGHT

        },

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
    # FINAL OUTPUT
    # ========================================================

    print("=" * 70)

    print(
        "TRUSTFED-6G TPRS EXPERIMENT COMPLETE"
    )

    print("=" * 70)

    final = round_results[-1]

    print()

    print(
        f"Final Accuracy  : "
        f"{final['accuracy']:.4f}"
    )

    print(
        f"Final Precision : "
        f"{final['precision']:.4f}"
    )

    print(
        f"Final Recall    : "
        f"{final['recall']:.4f}"
    )

    print(
        f"Final F1 Score  : "
        f"{final['f1']:.4f}"
    )

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()