cp -r data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training
rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/label/lidar
python tools/dataset_converters/label_world2v.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/label/lidar
python tools/dataset_converters/point_cloud_i2v.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne/lidar_i2v
python tools/dataset_converters/concatenate_pcd2bin.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --i2v-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne/lidar_i2v --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne-concated
rm -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne
mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne-concated ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/velodyne
python tools/dataset_converters/get_fusion_data_info.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training
rm ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/data_info.json
mv ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/fusion_data_info.json ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/data_info.json
python tools/dataset_converters/dair2kitti.py --source-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training --target-root ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training --split-path ./data/split_datas/cooperative-split-data.json --label-type lidar --sensor-view cooperative
python tools/create_data.py kitti --root-path data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/ --out-dir data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/ --extra-tag kitti
# cp -r ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/label/lidar ./data/DAIR-V2X/cooperative-vehicle-infrastructure/cooperative/
# cp data/DAIR-V2X/cooperative-vehicle-infrastructure/vic3d-early-fusion-training/training/velodyne/*  ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/velodyne
