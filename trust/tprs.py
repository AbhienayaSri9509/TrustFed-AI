def calculate_tprs(
    trust,
    quality,
    resource,
    network,
    privacy_suitability
):
    """
    Trust-Privacy-Resource Score (TPRS)

    TPRS combines:
    - client trust
    - local model/data quality
    - resource availability
    - network quality
    - privacy suitability
    """

    trust_weight = 0.30
    quality_weight = 0.25
    resource_weight = 0.20
    network_weight = 0.15
    privacy_weight = 0.10

    score = (

        trust_weight * trust

        + quality_weight * quality

        + resource_weight * resource

        + network_weight * network

        + privacy_weight * privacy_suitability

    )

    return float(score)


def rank_clients(client_scores):

    return sorted(
        client_scores,
        key=lambda x: x["tprs"],
        reverse=True
    )