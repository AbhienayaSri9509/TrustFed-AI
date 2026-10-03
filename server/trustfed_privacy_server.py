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

from privacy.adaptive_dp import (
    add_adaptive_noise,
    get_privacy_epsilon
)


# ============================================================
# TRUSTFED-6G
# TPRS + ADAPTIVE PRIVACY
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
    "TRUSTFED-6G TPRS + ADAPTIVE PRIVACY"
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
# CLIENT DATA
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

    metadata = pd.read_csv(
        path
    )

    return metadata


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
            0.0,
            1.0
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
            0.0,
            1.0
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
            1.0 / (
                1.0 + float(loss)
            ),
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

        for i in range(
            len(client_parameters)
        )

    )

    if total_weight <= 0:

        raise ValueError(
            "TPRS aggregation weight is zero."
        )

    aggregated = []

    for parameter_index in range(

        len(
            client_parameters[0]
        )

    ):

        result = None

        for client_index in range(

            len(
                client_parameters
            )

        ):

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

                client_parameters[
                    client_index
                ][
                    parameter_index
                ]
                *
                weight

            )

            if result is None:

                result = contribution

            else:

                result += contribution

        aggregated.append(
            result
        )

    return aggregated


# ============================================================
# GLOBAL TEST
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

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    print(
        "Loading client metadata..."
    )

    metadata = load_metadata()

    print(
        f"Metadata records: {len(metadata)}"
    )

    print()

    # --------------------------------------------------------
    # GLOBAL TEST
    # --------------------------------------------------------

    X_test, y_test = load_global_test()

    print(
        f"Global test samples: {len(X_test)}"
    )

    print(
        f"Feature dimension: {X_test.shape[1]}"
    )

    print()

    # --------------------------------------------------------
    # GLOBAL MODEL
    # --------------------------------------------------------

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
            f"TRUSTFED PRIVACY ROUND "
            f"{round_number}/{ROUNDS}"
        )

        print("=" * 70)

        local_results = []

        # ----------------------------------------------------
        # LOCAL TRAINING
        # ----------------------------------------------------

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
        # CLIENT SCORING
        # ----------------------------------------------------

        client_scores = []

        for result in local_results:

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
                1.0
                -
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

        selected = client_scores[
            :CLIENTS_PER_ROUND
        ]

        print()

        print(
            "TPRS + PRIVACY CLIENT RANKING:"
        )

        print(
            "-" * 85
        )

        print(
            "Client | Trust | Quality | "
            "Resource | Network | Privacy | "
            "Epsilon | TPRS"
        )

        print(
            "-" * 85
        )

        # ----------------------------------------------------
        # DISPLAY CLIENT SCORES
        # ----------------------------------------------------

        for client in client_scores:

            epsilon = get_privacy_epsilon(
                client[
                    "privacy_risk"
                ]
            )

            print(

                f"{client['client_id']:6d} | "

                f"{client['trust']:.3f} | "

                f"{client['quality']:.3f} | "

                f"{client['resource']:.3f} | "

                f"{client['network']:.3f} | "

                f"{client['privacy_risk']:.3f} | "

                f"{epsilon:.2f} | "

                f"{client['tprs']:.3f}"

            )

        selected_ids = [

            client[
                "client_id"
            ]

            for client
            in selected

        ]

        print()

        print(
            "Selected clients:"
        )

        print(
            selected_ids
        )

        # ----------------------------------------------------
        # ADAPTIVE PRIVACY
        # ----------------------------------------------------

        noisy_parameters = []

        selected_sizes = []

        selected_tprs = []

        privacy_records = []

        for client in selected:

            noisy, epsilon, noise_scale = (
                add_adaptive_noise(

                    client[
                        "parameters"
                    ],

                    client[
                        "privacy_risk"
                    ]

                )
            )

            noisy_parameters.append(
                noisy
            )

            selected_sizes.append(
                client[
                    "size"
                ]
            )

            selected_tprs.append(
                client[
                    "tprs"
                ]
            )

            privacy_records.append({

                "client_id":
                    client[
                        "client_id"
                    ],

                "privacy_risk":
                    client[
                        "privacy_risk"
                    ],

                "epsilon":
                    epsilon,

                "noise_scale":
                    noise_scale,

                "tprs":
                    client[
                        "tprs"
                    ]

            })

        # ----------------------------------------------------
        # PRIVATE TPRS AGGREGATION
        # ----------------------------------------------------

        print()

        print(
            "Applying adaptive privacy..."
        )

        global_parameters = tprs_aggregate(

            noisy_parameters,

            selected_sizes,

            selected_tprs

        )

        # ----------------------------------------------------
        # UPDATE GLOBAL MODEL
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
        # SAVE ROUND
        # ----------------------------------------------------

        round_results.append({

            "round":
                round_number,

            "selected_clients":
                selected_ids,

            "privacy":
                privacy_records,

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

        "trustfed_privacy_model.pth"

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

        "trustfed_privacy.json"

    )

    results = {

        "experiment":
            "TrustFed-6G TPRS + Adaptive Privacy",

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

        "privacy_method":
            "Adaptive experimental noise",

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

        "epsilon_policy": {

            "high_risk":
                0.5,

            "medium_risk":
                1.0,

            "low_risk":
                2.0

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
    # FINAL
    # ========================================================

    final = round_results[-1]

    print("=" * 70)

    print(
        "TRUSTFED-6G TPRS + ADAPTIVE PRIVACY COMPLETE"
    )

    print("=" * 70)

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