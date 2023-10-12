# Data

```shell
python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side \
    --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side \
    --split-path ./data/split_datas/cooperative-split-data.json \
    --label-type lidar --sensor-view infrastructure --no-classmerge

python tools/create_data.py kitti --root-path ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side --out-dir ./data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side --extra-tag kitti


python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side \
    --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side \
    --split-path ./data/split_datas/cooperative-split-data.json \
    --label-type lidar --sensor-view vehicle --no-classmerge

python tools/create_data.py kitti --root-path ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side --out-dir ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side --extra-tag kitti

```
