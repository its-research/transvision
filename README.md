# Transvision

## News

## CoFormerNet

### Results

| Car             | Latency | 3DAP@0.50 |   0.70    | BEVAP@0.50 |   0.70    |
| :-------------- | :-----: | :-------: | :-------: | :--------: | :-------: |
| FFNet-B-V       |   0ms   |   51.60   |   29.99   |   56.62    |   49.15   |
| FFNet-B-F       |   0ms   |   55.48   |   31.54   |   63.15    |   54.27   |
| FFNet           |   0ms   |   55.81   |   30.23   | __63.54__  |   54.16   |
| FFNet           |  200ms  |   55.37   |   31.66   |   63.20    | __54.69__ |
| FFNet(w/o pred) |  200ms  |   50.27   |   27.57   |   57.93    |   48.16   |
| TF-L-V          |   0ms   |   56.40   |   34.69   |   62.08    |   52.48   |
| TF-L-F          |   0ms   | __58.46__ | __37.28__ |   62.73    |   54.21   |
| CoFormerNet     |   0ms   |           |           |            |           |

conf=0.2

- FF-B-V: FFNet Basemodel veh-only(our re-implementation)
- FF-B-F: FFNet Basemodel fusion(our re-implementation)
- TF-L-V: Transfusion-L veh-only

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) commit: c65a55617f7d0a9b78dc9d107370c95bcac55dca
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D) commit:e2b7b7421efceb59d51f65b43c0f05bf031c3d0a
