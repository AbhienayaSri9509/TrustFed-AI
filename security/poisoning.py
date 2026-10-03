import numpy as np


def poison_parameters(
    global_parameters,
    local_parameters,
    attack_type="sign_flip",
    attack_strength=1.0
):
    """
    Apply a controlled poisoning attack to a client's
    locally trained model parameters.

    attack_type:
        sign_flip
        scaling
        gaussian

    Returns:
        poisoned model parameters
    """

    poisoned = []

    for global_param, local_param in zip(
        global_parameters,
        local_parameters
    ):
        update = (
            local_param - global_param
        )

        if attack_type == "sign_flip":

            poisoned_update = (
                -attack_strength * update
            )

        elif attack_type == "scaling":

            poisoned_update = (
                attack_strength * update
            )

        elif attack_type == "gaussian":

            noise = np.random.normal(
                0,
                attack_strength,
                size=update.shape
            ).astype(
                update.dtype
            )

            poisoned_update = (
                update + noise
            )

        else:

            raise ValueError(
                f"Unknown attack type: {attack_type}"
            )

        poisoned_parameter = (
            global_param +
            poisoned_update
        )

        poisoned.append(
            poisoned_parameter.astype(
                local_param.dtype
            )
        )

    return poisoned


def is_malicious(
    client_id,
    malicious_clients
):
    """
    Check whether a client is malicious.
    """

    return client_id in malicious_clients