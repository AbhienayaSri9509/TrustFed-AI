def update_trust(previous_trust,contribution_quality,beta=.7):
    return max(0,min(1,beta*previous_trust+(1-beta)*contribution_quality))
def penalize_for_anomaly(trust,anomaly_score,penalty=.3):
    return max(0,min(1,trust*(1-penalty*anomaly_score)))
