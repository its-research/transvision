# transvision

## Dependencies

- gcc==10
- CUDA==11.1
- python==3.8
- torch==1.9.1

```shell
pip install --upgrade git+https://github.com/klintan/pypcd.git
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.14 mmdet==2.14.0 mmsegmentation==0.14.1
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -v -e .

cd FFNET-VIC3D && pip install -v -e .
```

```shell
pip install -v -e .
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

```shell
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
```

## Issues

### 1. FileNotFoundError: \[Errno 2\] No such file or directory: './data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne/lidar_i2v/006315.pcd'

There is no '006315.pcd' file in cooperative-vehicle-infrastructure/infrastructure-side folders, we can just delete '006315.pcd''s imformation from cooperative-vehicle-infrastructure/cooperative/data_info.json

### 2. AttributeError: 'LineString' object has no attribute 'exterior'

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

### 3. kitti_annos\['name'\].append(label2cat\[label\]) KeyError: -1

mmdet3d/evaluation/metrics/kitti_metric.py

```python
label = instance['bbox_label']
if label == -1:
    continue
```

### 4. i15 = str(-eval(item\["rotation"\])) TypeError: eval() arg 1 must be a string, bytes or code object

```python
i15 = str(-eval(str(item["rotation"])))
```

### 5. commit

```bash
   pre-commit run --all-files
```

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) commit: cf92b8d9d91bd54bbdecc254550fcbc7c65b5dc7
- [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)
