# Tracking-atmospheric-phenomena
Master thesis project devoted to building tracks of atmospheric phenomena

## Requirements

This codebase is developed within the following environment:
```
python 3.5.2
tensorflow 1.13.1
tqdm 4.32.1
numpy 1.16.4
imgaug 0.4.0
```


## Trained models
| Model name      | Distance information| Neural Network Accuracy | MOTA |
|-----------------|---------------------|-------------------------|------|
| Random Mesocyclones| Not used | Not used  | -1.98 |
| Random Tropical Cyclones| Not used | Not used  | 0.04 |
| MCNNd| Used | 0.97±0.01 | 0.32±0.04 |
| MCNN| Not used | 0.88±0.01 | 0.29±0.06 |
| TCNN| Not used | 0.993±0.004 | 0.147±0.05 |

