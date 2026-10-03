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

from security.poisoning import (
    poison_parameters,
    is_malicious
)


# ============================================================
# TRUSTFED-6G
# E5 - POISONING ATTACK / ROBUSTNESS EXPERIMENT
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


# ------------------------------------------------------------
# CONTROLLED MALICIOUS CLIENTS
# ------------------------------------------------------------

MALICIOUS_CLIENTS = {
    0,
    3,
    7
}


ATTACK_TYPE = "sign_flip"

ATTACK_STRENGTH = 1.0


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "TRUSTFED-6G POISONING ATTACK EXPERIMENT"
)

print("=" * 70)

print(
    f"Total clients       : {NUM_CLIENTS}"
)

print(
    f"Clients per round   : {CLIENTS_PER_ROUND}"
)

print(
    f"Rounds              : {ROUNDS}"
)

print(
    f"Malicious clients   : "
    f"{sorted(MALICIOUS_CLIENTS)}"
)

print(
    f"Attack type         : {ATTACK_TYPE}"
)

print(
    f"Attack strength     : {ATTACK_STRENGTH}"
)

print()


# ============================================================
# MODEL
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

    return TensorDataset(
        X_tensor,
        y_tensor
    )


# ============================================================
# PARAMETERS
# ============================================================

def get_parameters(model):

    return [

        value.detach()
        .cpu()
        .numpy()
        .copy()

        for value
        in model.state_dict().values()

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

    model.load_state_dict(
        state_dict
    )


# ============================================================
# LOCAL TRAINING
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
# METADATA
# ============================================================

def load_metadata():

    path = os.path.join(
        CLIENT_DIR,
        "client_metadata.csv"
    )

    return pd.read_csv(
        path
    )


# ============================================================
# RESOURCE SCORE
# ============================================================

def resource_score(
    cpu,
    memory
):

    return float(
        np.clip(
            0.5 * float(cpu)
            +
            0.5 * float(memory),
            0,
            1
        )
    )


# ============================================================
# NETWORK SCORE
# ============================================================

def network_score(
    bandwidth,
    latency
):

    return float(
        np.clip(
            0.5 * float(bandwidth)
            +
            0.5 * float(latency),
            0,
            1
        )
    )


# ============================================================
# QUALITY SCORE
# ============================================================

def quality_score(
    loss
):

    return float(
        np.clip(
            1.0 /
            (
                1.0 +
                float(loss)
            ),
            0,
            1
        )
    )


# ============================================================
# FEDAVG
# ============================================================

def fedavg_aggregate(
    parameters,
    sizes
):

    total_size = sum(
        sizes
    )

    aggregated = []

    for parameter_index in range(
        len(parameters[0])
    ):

        result = np.zeros_like(
            parameters[0][
                parameter_index
            ],
            dtype=np.float32
        )

        for client_index in range(
            len(parameters)
        ):

            weight = (
                sizes[client_index]
                /
                total_size
            )

            result += (
                parameters[
                    client_index
                ][
                    parameter_index
                ]
                * weight
            )

        aggregated.append(
            result
        )

    return aggregated


# ============================================================
# TPRS AGGREGATION
# ============================================================

def tprs_aggregate(
    parameters,
    sizes,
    tprs
):

    total_weight = sum(

        tprs[i]
        *
        sizes[i]

        for i in range(
            len(parameters)
        )

    )

    aggregated = []

    for parameter_index in range(
        len(parameters[0])
    ):

        result = np.zeros_like(
            parameters[0][
                parameter_index
            ],
            dtype=np.float32
        )

        for client_index in range(
            len(parameters)
        ):

            weight = (

                tprs[client_index]
                *
                sizes[client_index]
                /
                total_weight

            )

            result += (

                parameters[
                    client_index
                ][
                    parameter_index
                ]
                *
                weight

            )

        aggregated.append(
            result
        )

    return aggregated


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_global_test():

    path = os.path.join(
        CLIENT_DIR,
        "global_test.npz"
    )

    data = np.load(
        path
    )

    return (

        data["X"].astype(
            np.float32
        ),

        data["y"].astype(
            np.float32
        )

    )


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

        probabilities = torch.sigmoid(
            output
        ).numpy().reshape(
            -1
        )

    predictions = (
        probabilities >= 0.5
    ).astype(
        int
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    return {

        "accuracy":
            float(
                accuracy_score(
                    y,
                    predictions
                )
            ),

        "precision":
            float(
                precision_score(
                    y,
                    predictions,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y,
                    predictions,
                    zero_division=0
                )
            ),

        "f1":
            float(
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
# MAIN
# ============================================================

def main():

    metadata = load_metadata()

    X_test, y_test = load_global_test()

    global_model = MLP(
        input_dim=X_test.shape[1]
    )

    global_parameters = get_parameters(
        global_model
    )

    all_rounds = []

    # ========================================================
    # FEDERATED TRAINING
    # ========================================================

    for round_number in range(
        1,
        ROUNDS + 1
    ):

        print("=" * 70)

        print(
            f"POISONING ROUND "
            f"{round_number}/{ROUNDS}"
        )

        print("=" * 70)

        client_results = []

        # ----------------------------------------------------
        # TRAIN CLIENTS
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

            malicious = is_malicious(
                client_id,
                MALICIOUS_CLIENTS
            )

            # ------------------------------------------------
            # APPLY ATTACK
            # ------------------------------------------------

            if malicious:

                print(
                    f"Client {client_id}: "
                    f"MALICIOUS "
                    f"-> applying "
                    f"{ATTACK_TYPE}"
                )

                parameters = (
                    poison_parameters(

                        global_parameters,

                        parameters,

                        attack_type=
                            ATTACK_TYPE,

                        attack_strength=
                            ATTACK_STRENGTH

                    )
                )

            client_results.append({

                "client_id":
                    client_id,

                "parameters":
                    parameters,

                "size":
                    size,

                "loss":
                    loss,

                "malicious":
                    malicious

            })

        # ----------------------------------------------------
        # TPRS SCORING
        # ----------------------------------------------------

        scored_clients = []

        for result in client_results:

            client_id = result[
                "client_id"
            ]

            row = metadata[
                metadata[
                    "client_id"
                ]
                ==
                client_id
            ].iloc[0]

            trust = float(
                row["trust"]
            )

            quality = quality_score(
                result["loss"]
            )

            resource = resource_score(

                row[
                    "cpu_available"
                ],

                row[
                    "memory_available"
                ]

            )

            network = network_score(

                row[
                    "bandwidth"
                ],

                row[
                    "latency_score"
                ]

            )

            privacy_risk = float(
                row[
                    "privacy_risk"
                ]
            )

            privacy_suitability = (
                1.0 -
                privacy_risk
            )

            tprs = calculate_tprs(

                trust=trust,

                quality=quality,

                resource=resource,

                network=network,

                privacy_suitability=
                    privacy_suitability

            )

            scored_clients.append({

                "client_id":
                    client_id,

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
                    ],

                "malicious":
                    result[
                        "malicious"
                    ],

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
                    tprs

            })

        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        scored_clients.sort(

            key=lambda x:
                x["tprs"],

            reverse=True

        )

        selected = scored_clients[
            :CLIENTS_PER_ROUND
        ]

        selected_ids = [

            client[
                "client_id"
            ]

            for client
            in selected

        ]

        selected_malicious = [

            client[
                "client_id"
            ]

            for client
            in selected

            if client[
                "malicious"
            ]

        ]

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print()

        print(
            "CLIENT TPRS RANKING"
        )

        print(
            "-" * 75
        )

        print(
            "Client | Malicious | Trust | "
            "Quality | TPRS"
        )

        print(
            "-" * 75
        )

        for client in scored_clients:

            print(

                f"{client['client_id']:6d} | "

                f"{str(client['malicious']):9s} | "

                f"{client['trust']:.3f} | "

                f"{client['quality']:.3f} | "

                f"{client['tprs']:.3f}"

            )

        print()

        print(
            f"Selected clients: "
            f"{selected_ids}"
        )

        print(
            f"Selected malicious clients: "
            f"{selected_malicious}"
        )

        # ----------------------------------------------------
        # AGGREGATION
        # ----------------------------------------------------

        selected_parameters = [

            client[
                "parameters"
            ]

            for client
            in selected

        ]

        selected_sizes = [

            client[
                "size"
            ]

            for client
            in selected

        ]

        selected_tprs = [

            client[
                "tprs"
            ]

            for client
            in selected

        ]

        global_parameters = (
            tprs_aggregate(

                selected_parameters,

                selected_sizes,

                selected_tprs

            )
        )

        # ----------------------------------------------------
        # UPDATE MODEL
        # ----------------------------------------------------

        set_parameters(

            global_model,

            global_parameters

        )

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        metrics = evaluate(

            global_model,

            X_test,

            y_test

        )

        print()

        print(
            "Round Results"
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
        # SAVE ROUND
        # ----------------------------------------------------

        all_rounds.append({

            "round":
                round_number,

            "selected_clients":
                selected_ids,

            "selected_malicious_clients":
                selected_malicious,

            "num_selected_malicious":
                len(
                    selected_malicious
                ),

            "metrics":
                metrics,

            "client_scores": [

                {

                    "client_id":
                        client[
                            "client_id"
                        ],

                    "malicious":
                        client[
                            "malicious"
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
                in scored_clients

            ]

        })

        print()

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(

        MODELS_DIR,

        "trustfed_attack_model.pth"

    )

    torch.save(

        global_model.state_dict(),

        model_path

    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "experiment":
            "TrustFed-6G Poisoning Attack",

        "dataset":
            "UNSW-NB15",

        "num_clients":
            NUM_CLIENTS,

        "rounds":
            ROUNDS,

        "clients_per_round":
            CLIENTS_PER_ROUND,

        "malicious_clients":
            sorted(
                MALICIOUS_CLIENTS
            ),

        "attack_type":
            ATTACK_TYPE,

        "attack_strength":
            ATTACK_STRENGTH,

        "rounds_results":
            all_rounds,

        "final_results":
            all_rounds[-1]

    }

    metrics_path = os.path.join(

        METRICS_DIR,

        "trustfed_attack.json"

    )

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

    final_metrics = (
        all_rounds[-1][
            "metrics"
        ]
    )

    print("=" * 70)

    print(
        "TRUSTFED-6G POISONING EXPERIMENT COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Final Accuracy  : "
        f"{final_metrics['accuracy']:.4f}"
    )

    print(
        f"Final Precision : "
        f"{final_metrics['precision']:.4f}"
    )

    print(
        f"Final Recall    : "
        f"{final_metrics['recall']:.4f}"
    )

    print(
        f"Final F1 Score  : "
        f"{final_metrics['f1']:.4f}"
    )

    print()

    print(
        "Model:"
    )

    print(
        model_path
    )

    print()

    print(
        "Metrics:"
    )

    print(
        metrics_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()