import numpy as np
def normalized_tprs_weights(records):
    raw=np.array([max(0,r.get('tprs',0))*max(0,r.get('data_quality',.5)) for r in records])
    return raw/raw.sum() if raw.sum() else np.ones(len(raw))/len(raw)
def weighted_average(values,weights):
    result=np.zeros_like(values[0],dtype=float)
    for v,w in zip(values,weights): result+=w*v
    return result
