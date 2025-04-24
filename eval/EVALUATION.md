# Evaluation

## Experiment settings
  - `input_size = 784`
  - `layer1_dim = 320`
  - `layer2_dim = 160`
  - `layer3_dim = 10`
  - `block_size = 16`
  - `TILE_SIZE = 16`
  - `batch_size ∈ {32, 64, 96, 128}`
  - `lr ∈ {0.03, 0.03, 0.04, 0.04}`
---

## Time Per Epoch
<img src=../images/tpe.png width="700px">

## GPU Utilization
<img src=../images/gpu_util.png width="700px">

## Validation Accuracy
<img src=../images/acc.png width="700px">