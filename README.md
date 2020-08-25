# Tracking-atmospheric-phenomena
Master thesis project devoted to building tracks of atmospheric phenomena.

## Run models

In order to run tcnn model to build tracks. Also it's possible to run following tracking models: mcnn, mcnnd, random_tc, random_mc.

```bash
python track.py tcnn
```

In order to train mcnn model. Also it's possible to train 'mcnnd' and 'tcnn' models. For 'tcnn' model input shape is (80,80,1).

```bash
python train.py --model=mcnn --input_shape='200,200,2' --epochs=300 --train
```

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
| Random Mesocyclones| Not used | Not used  | -3.19±0.25 |
| Random Tropical Cyclones| Not used | Not used  | 0.0569±0.0015  |
| MCNNd| Used | 0.97±0.01 | 0.32±0.04 |
| MCNN| Not used | 0.88±0.01 | 0.29±0.06 |
| TCNN| Not used | 0.993±0.004 | 0.147±0.05 |

