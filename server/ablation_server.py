import os
import json
import random

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
from privacy.adaptive_dp import (
    add_adaptive_noise,
    get_privacy_epsilon
)


# ============================================================
# TRUSTFED-6G
# E6 - ABLATION STUDY
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
# EXPERIMENT CONFIGURATION
# ============================================================

NUM_CLIENTS = 10
ROUNDS = 5
CLIENTS_PER_ROUND = 7

LOCAL_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.001

INPUT_DIM = 192

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

EXPERIMENTS = {

    "A1_FedAvg": {
        "trust": False,
        "resource": False,
        "network": False,
        "privacy": False,
        "tprs": False
    },

    "A2_Trust": {
        "trust": True,
        "resource": False,
        "network": False,
        "privacy": False,
        "tprs": False
    },

    "A3_Trust_Resource": {
        "trust": True,
        "resource": True,
        "network": False,
        "privacy": False,
        "tprs": False
    },

    "A4_Trust_Resource_Network": {
        "trust": True,
        "resource": True,
        "network": True,
        "privacy": False,
        "tprs": False
    },

    "A5_TPRS": {
        "trust": True,
        "resource": True,
        "network": True,
        "privacy": True,
        "tprs": True
    },

    "A6_Full_TrustFed": {
        "trust": True,
        "resource": True,
        "network": True,
        "privacy": True,
        "tprs": True,
        "adaptive_privacy": True
    }
}


# ============================================================
# MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_dim=192):

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

    def forward(self, x):

        return self.network(x)


# ============================================================
# PARAMETERS
# ============================================================

def get_parameters(model):

    return [

        value.detach()
        .cpu()
        .numpy()
        .copy()

        for value in model.state_dict().values()

    ]


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

    model.load_state_dict(state_dict)


# ============================================================
# DATA
# ============================================================

def load_client(client_id):

    path = os.path.join(
        CLIENT_DIR,
        f"client_{client_id}.npz"
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

    return TensorDataset(
        X_tensor,
        y_tensor
    )


def load_global_test():

    path = os.path.join(
        CLIENT_DIR,
        "global_test.npz"
    )

    data = np.load(path)

    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.float32)
    )


def load_metadata():

    path = os.path.join(
        CLIENT_DIR,
        "client_metadata.csv"
    )

    return pd.read_csv(path)


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_client(
    client_id,
    global_parameters
):

    dataset = load_client(client_id)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = MLP(INPUT_DIM)

    set_parameters(
        model,
        global_parameters
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    total_loss = 0.0

    for _ in range(LOCAL_EPOCHS):

        epoch_loss = 0.0

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            output = model(X_batch)

            loss = criterion(
                output,
                y_batch
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        total_loss = (
            epoch_loss /
            max(len(loader), 1)
        )

    return (
        get_parameters(model),
        len(dataset),
        total_loss
    )


# ============================================================
# SCORE FUNCTIONS
# ============================================================

def quality_score(loss):

    return float(
        np.clip(
            1.0 / (1.0 + float(loss)),
            0.0,
            1.0
        )
    )


def resource_score(
    cpu,
    memory
):

    return float(
        np.clip(
            0.5 * float(cpu)
            +
            0.5 * float(memory),
            0.0,
            1.0
        )
    )


def network_score(
    bandwidth,
    latency
):

    return float(
        np.clip(
            0.5 * float(bandwidth)
            +
            0.5 * float(latency),
            0.0,
            1.0
        )
    )


# ============================================================
# CLIENT WEIGHT
# ============================================================

def calculate_client_weight(
    client,
    config
):

    weights = []

    # --------------------------------------------------------
    # Trust
    # --------------------------------------------------------

    if config["trust"]:

        weights.append(
            0.30 *
            client["trust"]
        )

    # --------------------------------------------------------
    # Resource
    # --------------------------------------------------------

    if config["resource"]:

        weights.append(
            0.20 *
            client["resource"]
        )

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    if config["network"]:

        weights.append(
            0.15 *
            client["network"]
        )

    # --------------------------------------------------------
    # Privacy suitability
    # --------------------------------------------------------

    if config["privacy"]:

        privacy_suitability = (
            1.0 -
            client["privacy_risk"]
        )

        weights.append(
            0.10 *
            privacy_suitability
        )

    # --------------------------------------------------------
    # TPRS
    # --------------------------------------------------------

    if config["tprs"]:

        return calculate_tprs(

            trust=client["trust"],

            quality=client["quality"],

            resource=client["resource"],

            network=client["network"],

            privacy_suitability=(
                1.0 -
                client["privacy_risk"]
            )

        )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    if not weights:

        return 1.0

    # Add data quality to trust/resource/network variants
    quality_component = (
        0.25 *
        client["quality"]
    )

    return float(
        sum(weights)
        +
        quality_component
    )


# ============================================================
# AGGREGATION
# ============================================================

def aggregate(
    parameters,
    sizes,
    weights
):

    total_weight = sum(

        weights[i] *
        sizes[i]

        for i in range(
            len(parameters)
        )

    )

    if total_weight <= 0:

        raise ValueError(
            "Aggregation weight is zero."
        )

    aggregated = []

    for parameter_index in range(
        len(parameters[0])
    ):

        result = np.zeros_like(
            parameters[0][parameter_index],
            dtype=np.float32
        )

        for i in range(
            len(parameters)
        ):

            contribution_weight = (

                weights[i] *
                sizes[i] /
                total_weight

            )

            result += (

                parameters[i][
                    parameter_index
                ]
                *
                contribution_weight

            )

        aggregated.append(result)

    return aggregated


# ============================================================
# EVALUATION
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

        probabilities = (
            torch.sigmoid(output)
            .numpy()
            .reshape(-1)
        )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        y,
        predictions
    )

    return {

        "accuracy": float(
            accuracy_score(
                y,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0
            )
        ),

        "confusion_matrix":
            cm.tolist()

    }


# ============================================================
# RUN ONE ABLATION
# ============================================================

def run_experiment(
    experiment_name,
    config,
    metadata,
    X_test,
    y_test
):

    print()
    print("=" * 75)
    print(
        f"RUNNING: {experiment_name}"
    )
    print("=" * 75)

    model = MLP(INPUT_DIM)

    global_parameters = get_parameters(model)

    rounds = []

    for round_number in range(
        1,
        ROUNDS + 1
    ):

        print(
            f"\nRound "
            f"{round_number}/{ROUNDS}"
        )

        client_results = []

        # ----------------------------------------------------
        # TRAIN ALL CLIENTS
        # ----------------------------------------------------

        for client_id in range(
            NUM_CLIENTS
        ):

            parameters, size, loss = (
                train_client(
                    client_id,
                    global_parameters
                )
            )

            row = metadata[
                metadata["client_id"]
                ==
                client_id
            ].iloc[0]

            client = {

                "client_id":
                    client_id,

                "parameters":
                    parameters,

                "size":
                    size,

                "loss":
                    loss,

                "trust":
                    float(row["trust"]),

                "quality":
                    quality_score(loss),

                "resource":
                    resource_score(
                        row["cpu_available"],
                        row["memory_available"]
                    ),

                "network":
                    network_score(
                        row["bandwidth"],
                        row["latency_score"]
                    ),

                "privacy_risk":
                    float(
                        row["privacy_risk"]
                    )

            }

            client["weight"] = (
                calculate_client_weight(
                    client,
                    config
                )
            )

            client_results.append(client)

        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        if config["tprs"]:

            client_results.sort(
                key=lambda x:
                    x["weight"],
                reverse=True
            )

        else:

            client_results.sort(
                key=lambda x:
                    x["client_id"]
            )

        selected = client_results[
            :CLIENTS_PER_ROUND
        ]

        selected_ids = [
            c["client_id"]
            for c in selected
        ]

        # ----------------------------------------------------
        # ADAPTIVE PRIVACY
        # ----------------------------------------------------

        epsilons = []

        selected_parameters = []

        selected_sizes = []

        selected_weights = []

        for client in selected:

            parameters = client[
                "parameters"
            ]

            epsilon = None

            if config.get(
                "adaptive_privacy",
                False
            ):

                (
                    parameters,
                    epsilon,
                    _
                ) = add_adaptive_noise(

                    parameters,

                    client[
                        "privacy_risk"
                    ]

                )

                epsilons.append(
                    epsilon
                )

            selected_parameters.append(
                parameters
            )

            selected_sizes.append(
                client["size"]
            )

            selected_weights.append(
                client["weight"]
            )

        # ----------------------------------------------------
        # AGGREGATE
        # ----------------------------------------------------

        global_parameters = aggregate(

            selected_parameters,

            selected_sizes,

            selected_weights

        )

        set_parameters(
            model,
            global_parameters
        )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        metrics = evaluate(
            model,
            X_test,
            y_test
        )

        avg_epsilon = None

        if epsilons:

            avg_epsilon = float(
                np.mean(epsilons)
            )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f} | "
            f"Precision: "
            f"{metrics['precision']:.4f} | "
            f"Recall: "
            f"{metrics['recall']:.4f} | "
            f"F1: "
            f"{metrics['f1']:.4f}"
        )

        rounds.append({

            "round":
                round_number,

            "selected_clients":
                selected_ids,

            "average_epsilon":
                avg_epsilon,

            "metrics":
                metrics

        })

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    safe_name = (
        experiment_name
        .lower()
        .replace(" ", "_")
    )

    model_path = os.path.join(
        MODELS_DIR,
        f"ablation_{safe_name}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    final = rounds[-1]

    return {

        "experiment":
            experiment_name,

        "configuration":
            config,

        "rounds":
            rounds,

        "final_accuracy":
            final["metrics"]["accuracy"],

        "final_precision":
            final["metrics"]["precision"],

        "final_recall":
            final["metrics"]["recall"],

        "final_f1":
            final["metrics"]["f1"],

        "average_final_epsilon":
            final["average_epsilon"]

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TRUSTFED-6G E6 ABLATION STUDY")
    print("=" * 75)

    print(
        f"Clients : {NUM_CLIENTS}"
    )

    print(
        f"Rounds  : {ROUNDS}"
    )

    print(
        f"Selected per round : "
        f"{CLIENTS_PER_ROUND}"
    )

    print()

    metadata = load_metadata()

    X_test, y_test = load_global_test()

    results = {}

    # ========================================================
    # RUN ALL CONFIGURATIONS
    # ========================================================

    for experiment_name, config in (
        EXPERIMENTS.items()
    ):

        results[
            experiment_name
        ] = run_experiment(

            experiment_name,

            config,

            metadata,

            X_test,

            y_test

        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = os.path.join(
        METRICS_DIR,
        "ablation_results.json"
    )

    with open(
        json_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    rows = []

    for name, result in (
        results.items()
    ):

        rows.append({

            "Experiment":
                name,

            "Accuracy":
                result[
                    "final_accuracy"
                ],

            "Precision":
                result[
                    "final_precision"
                ],

            "Recall":
                result[
                    "final_recall"
                ],

            "F1":
                result[
                    "final_f1"
                ],

            "Avg_Epsilon":
                result[
                    "average_final_epsilon"
                ]

        })

    dataframe = pd.DataFrame(rows)

    csv_path = os.path.join(
        METRICS_DIR,
        "ablation_results.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL ABLATION RESULTS")
    print("=" * 75)

    print()

    print(
        dataframe.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()

    print("=" * 75)

    print(
        "ABLATION STUDY COMPLETE"
    )

    print("=" * 75)

    print()

    print(
        "JSON:"
    )

    print(
        json_path
    )

    print()

    print(
        "CSV:"
    )

    print(
        csv_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()