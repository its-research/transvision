# transvision

## Dependencies

+ mmdetection3d==v1.1.0
+ mmcv==2.0.0
+ mmdet==3.0.0

```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
mim install "mmdet3d>=1.1.0"
```

## Data Preparation

### DAIR-V2X-V

```shell
python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/single-vehicle-side --target-root ./data/DAIR-V2X/single-vehicle-side --split-path ./data/split_datas/single-vehicle-split-data.json --label-type camera --sensor-view vehicle

python tools/create_data.py kitti --root-path data/DAIR-V2X/single-vehicle-side/ --out-dir data/DAIR-V2X/single-vehicle-side/ --extra-tag kitti
```

### DAIR-V2X-I

```shell
python tools/dataset_converter/dair2kitti.py --source-root ./data/DAIR-V2X/single-infrastructure-side --target-root ./data/DAIR-V2X/single-infrastructure-side --split-path ./data/split_datas/single-infrastructure-split-data.json --label-type camera --sensor-view infrastructure

python tools/create_data.py kitti --root-path data/DAIR-V2X/single-infrastructure-side --out-dir data/DAIR-V2X/single-infrastructure-side --extra-tag kitti
```

### DAIR-V2X-C

```shell
python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side --split-path ./data/split_datas/cooperative-split-data.json --label-type camera --sensor-view infrastructure --no-classmerge

python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side --split-path ./data/split_datas/cooperative-split-data.json --label-type camera --sensor-view vehicle --no-classmerge

python tools/create_data.py kitti --root-path ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side  --out-dir ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side  --extra-tag kitti

python tools/create_data.py kitti --root-path ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side --out-dir ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side --extra-tag kitti
```

## Reference

+ [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
+ [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
+ [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)