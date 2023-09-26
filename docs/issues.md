# Issues

## 1. FileNotFoundError: \[Errno 2\] No such file or directory: './data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne/lidar_i2v/006315.pcd'

There is no '006315.pcd' file in cooperative-vehicle-infrastructure/infrastructure-side folders, we can just delete '006315.pcd''s imformation from cooperative-vehicle-infrastructure/cooperative/data_info.json

## 2. AttributeError: 'LineString' object has no attribute 'exterior'

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

## 3. kitti_annos\['name'\].append(label2cat\[label\]) KeyError: -1

mmdet3d/evaluation/metrics/kitti_metric.py

```python
label = instance['bbox_label']
if label == -1:
    continue
```

## 4. i15 = str(-eval(item\["rotation"\])) TypeError: eval() arg 1 must be a string, bytes or code object

```python
i15 = str(-eval(str(item["rotation"])))
```

## 5. commit

```bash
   pre-commit run --all-files
```

### 6. AttributeError: 'str' object has no attribute 'new_zeros'

undefine loss function
