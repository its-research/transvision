# Transvision

## News

## CoFormerNet

### Results

#### Veh-only

| Car    | 3DAP@0.30 | 0.50  | 0.70  | BEVAP@0.30 | 0.50  | 0.70  |
| :----- | :-------: | :---: | :---: | :--------: | :---: | :---: |
| TF-L-V |   71.03   | 57.88 | 27.41 |   73.76    | 66.63 | 51.26 |
| FF-B-V |   62.37   | 53.40 | 30.25 |   64.10    | 59.57 | 50.36 |
| FFNet  |   63.55   | 55.48 | 31.54 |   65.67    | 63.15 | 54.27 |

TF-L-V: Transfusion-L veh-only
FF-B-V: FFNet Basemodel veh-only (conf=0.02)
FF: FFNet

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) commit: cf92b8d9d91bd54bbdecc254550fcbc7c65b5dc7
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D) commit:e2b7b7421efceb59d51f65b43c0f05bf031c3d0a
- [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)
