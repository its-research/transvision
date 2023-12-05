# Transvision

## News

## CoFormerNet

### Results

| Car         | 3DAP@0.30 | 0.50  | 0.70  | BEVAP@0.30 | 0.50  | 0.70  |
| :---------- | :-------: | :---: | :---: | :--------: | :---: | :---: |
| FFNet-B-V   |   58.70   | 51.60 | 29.99 |   59.87    | 56.62 | 49.15 |
| FFNet-B-F   |   63.55   | 55.48 | 31.54 |   65.67    | 63.15 | 54.27 |
| FFNet-100ms |   69.91   | 60.97 | 33.97 |   71.81    | 69.13 | 58.65 |
| FFNet-200ms |   69.35   | 60.63 | 34.14 |   71.22    | 68.60 | 58.29 |
| TF-L-V      |   64.91   | 56.40 | 34.69 |   66.21    | 62.08 | 52.48 |
| CoFormerNet |           |       |       |            |       |       |

conf=0.2

- FF-B-V: FFNet Basemodel veh-only(our re-implementation)
- FF-B-F: FFNet Basemodel fusion(our re-implementation)
- FFNet-100ms: FFNet with latency: 100ms(our re-implementation)
- FFNet-200ms: FFNet with latency: 200ms(our re-implementation)
- TF-L-V: Transfusion-L veh-only

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) commit: c65a55617f7d0a9b78dc9d107370c95bcac55dca
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D) commit:e2b7b7421efceb59d51f65b43c0f05bf031c3d0a
