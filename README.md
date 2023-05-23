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
bash ./scripts/convert_dair_v2x_c.sh
```

05/22 18:27:38 - mmengine - INFO - The number of instances per category in the dataset:
+----------------+--------+
| category       | number |
+----------------+--------+
| Pedestrian     | 0      |
| Cyclist        | 0      |
| Car            | 65920  |
| Van            | 8231   |
| Truck          | 4100   |
| Person_sitting | 0      |
| Tram           | 0      |
| Misc           | 0      |
+----------------+--------+
Car            | 81393


## Issues

1. FileNotFoundError: [Errno 2] No such file or directory: './data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne/lidar_i2v/006315.pcd'

There is no '006315.pcd' file in cooperative-vehicle-infrastructure/infrastructure-side folders, we can just delete '006315.pcd''s imformation from cooperative-vehicle-infrastructure/cooperative/data_info.json

2. AttributeError: 'LineString' object has no attribute 'exterior'

```python
from shapely.geometry.polygon import Polygon
if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        if isinstance(img_intersection, Polygon):
            intersection_coords = np.array(
                [coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None
else:
	return None
```

## Reference

+ [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
+ [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
+ [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)