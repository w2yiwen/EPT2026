*Table 2. Performance comparison on the fixed test evaluation view (%).*

| Method | AUPRC ↑ | AP@0.3 ↑ | AP@0.5 ↑ | AP@0.7 ↑ | Event-F1@0.5 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTR | 86.95 | 39.75 | 32.94 | 22.65 | 67.44 |
| GateHUB | 94.63 | 35.90 | 32.91 | 28.54 | 75.61 |
| TeSTra | 96.56 | **41.43** | **38.77** | **29.43** | 74.36 |
| **EPT-Net (ours)** | **97.00** | 29.18 | 27.35 | 18.87 | **76.71** |
| *Mechanism diagnostics* |  |  |  |  |  |
| Fixed reader | 97.12 | 30.47 | 29.09 | 22.50 | 82.05 |
| No persistent state | 98.23 | 60.90 | 60.90 | 53.89 | 88.37 |

Seeds: 42 (n=1). One run is available, so values are point estimates without a standard deviation.
Bold marks the best value within the four-model main comparison only. 
Fixed reader and No persistent state are mechanism diagnostics, not additional main baselines. 
The project's validation and test manifests are fixed training-included evaluation views, not held-out splits.
