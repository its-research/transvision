# mkdir data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training
cp -r data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training
rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/label/lidar
python tools/dataset_converters/label_world2v.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/label/lidar

# python tools/dataset_converters/point_cloud_i2v.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/velodyne/lidar_i2v
# python tools/dataset_converters/concatenate_pcd2bin.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --i2v-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/velodyne/lidar_i2v --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/velodyne-concated
# rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/velodyne
# mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/velodyne-concated ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/velodyne

python tools/dataset_converters/get_fusion_data_info.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training
rm ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/data_info.json
mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/fusion_data_info.json ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/data_info.json
python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training --split-path ./data/split_datas/cooperative-split-data.json --label-type lidar --sensor-view cooperative --no-classmerge
python tools/create_data.py kitti --root-path data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/ --out-dir data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_0.17.1_training/ --extra-tag kitti
