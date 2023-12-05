# Transvision

## News

## CoFormerNet

### Results

| Car    | 3DAP@0.30 | 0.50  | 0.70  | BEVAP@0.30 | 0.50  | 0.70  |
| :----- | :-------: | :---: | :---: | :--------: | :---: | :---: |
| TF-L-V |   64.91   | 56.40 | 34.69 |   66.21    | 62.08 | 52.48 |
| FF-B-V |   58.70   | 51.60 | 29.99 |   59.87    | 56.62 | 49.15 |
| FFNet  |   63.55   | 55.48 | 31.54 |   65.67    | 63.15 | 54.27 |

conf=0.2
TF-L-V: Transfusion-L veh-only
FF-B-V: FFNet Basemodel veh-only
FF: FFNet(Ours, latency: 100ms)

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) commit: c65a55617f7d0a9b78dc9d107370c95bcac55dca
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D) commit:e2b7b7421efceb59d51f65b43c0f05bf031c3d0a
- [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)
