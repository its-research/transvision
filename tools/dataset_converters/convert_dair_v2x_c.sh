cp -r data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion
rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/label/lidar
python tools/dataset_converters/label_world2v.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/label/lidar

# python tools/dataset_converters/point_cloud_i2v.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/velodyne/lidar_i2v
# python tools/dataset_converters/concatenate_pcd2bin.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --i2v-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/velodyne/lidar_i2v --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/velodyne-concated
# rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/velodyne
# mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/velodyne-concated ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/velodyne

python tools/dataset_converters/get_fusion_data_info.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion
rm ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/data_info.json
mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/fusion_data_info.json ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/data_info.json
python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion --split-path ./data/split_datas/cooperative-split-data.json --label-type lidar --sensor-view cooperative --no-classmerge
python tools/create_data.py kitti --root-path data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/ --out-dir data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/ --extra-tag kitti

# cp -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/label/lidar ./data/DAIR-V2X/cooperative-vehicle-infrastructure/cooperative/
# cp data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/training/velodyne/*  ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/velodyne

# rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/cooperative/label/lidar
# cp -r data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/label/lidar ./data/DAIR-V2X/cooperative-vehicle-infrastructure/cooperative/label/
# cp data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/early-fusion/training/velodyne/*.bin data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/velodyne/
