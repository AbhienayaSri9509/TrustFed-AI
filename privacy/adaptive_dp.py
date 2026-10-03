import numpy as np


# ============================================================
# TRUSTFED-6G ADAPTIVE PRIVACY
# ============================================================

def get_privacy_epsilon(privacy_risk):
    """
    Select experimental privacy strength based on privacy risk.

    Higher privacy risk -> stronger protection -> smaller epsilon.
    """

    privacy_risk = float(
        np.clip(
            privacy_risk,
            0.0,
            1.0
        )
    )

    if privacy_risk >= 0.70:
        return 0.5

    elif privacy_risk >= 0.40:
        return 1.0

    else:
        return 2.0


def get_noise_scale(epsilon, sensitivity=1.0):
    """
    Experimental noise scale.

    This is an adaptive noise mechanism.
    It should NOT be interpreted as a formal DP guarantee
    without a complete privacy accountant.
    """

    epsilon = max(
        float(epsilon),
        1e-8
    )

    return (
        sensitivity / epsilon
    )


def add_adaptive_noise(
    parameters,
    privacy_risk,
    sensitivity=1.0
):
    """
    Add adaptive Gaussian noise to model parameters.

    Returns:
        noisy parameters
        epsilon
        noise scale
    """

    epsilon = get_privacy_epsilon(
        privacy_risk
    )

    noise_scale = get_noise_scale(
        epsilon,
        sensitivity
    )

    noisy_parameters = []

    for parameter in parameters:

        noise = np.random.normal(
            loc=0.0,
            scale=noise_scale * 0.01,
            size=parameter.shape
        ).astype(
            parameter.dtype
        )

        noisy_parameter = (
            parameter + noise
        )

        noisy_parameters.append(
            noisy_parameter
        )

    return (
        noisy_parameters,
        epsilon,
        noise_scale
    )