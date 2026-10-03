# Structure
data/raw: uploaded datasets
data/processed: cleaned data
data/client_splits: client partitions/metadata
model: MLP + data loading
clients: virtual client creation
server: FedAvg + TPRS selection/aggregation
trust: TPRS + trust update
privacy: adaptive DP
security: poisoning experiments
utils: metrics/seeding
configs: experiment settings
results: plots/metrics
