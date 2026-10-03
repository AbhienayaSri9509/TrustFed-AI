# Implementation Plan
1. Inspect UNSW-NB15 and confirm columns/target.
2. Preprocess and encode/scale.
3. Train centralized MLP/DNN baseline.
4. Partition training data into heterogeneous virtual clients.
5. Implement Flower FedAvg.
6. Calculate TPRS each round and select clients.
7. Add adaptive DP and measure privacy/utility.
8. Add TPRS-weighted aggregation.
9. Run controlled label/model poisoning.
10. Compare accuracy, F1, attack detection, communication and convergence.
