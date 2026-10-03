import pandas as pd
from trust.tprs import rank_clients
df=pd.read_csv('data/client_splits/client_metadata.csv'); clients=df.to_dict('records')
for c in clients:c['data_quality']=.5
for c in rank_clients(clients):print(f"Client {c['client_id']}: TPRS={c['tprs']:.3f}")
