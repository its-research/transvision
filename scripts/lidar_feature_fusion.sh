DATA='data/DAIR-V2X/cooperative-vehicle-infrastructure'
SPLIT=val
SPLIT_DATA_PATH="data/split_datas/cooperative-split-data.json"
OUTPUT="cache/vic-feature-fusion-baseline"
# VEHICLE_MODEL_PATH='work_dirs/mmdet3d_1.3.0/ffnet-vic3d/basemodel/fusion/lr=0.0036/epoch_30.pth'
# VEHICLE_CONFIG_NAME='configs/ffnet/config_basemodel_fusion.py'
VEHICLE_MODEL_PATH='work_dirs/mmdet3d_1.3.0/ffnet-vic3d/basemodel/veh_only/epoch_40.pth'
VEHICLE_CONFIG_NAME='configs/ffnet/config_basemodel_veh_only.py'
CUDA_VISIBLE_DEVICES=$1 \

python transvision/eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model feature_fusion \
  --dataset vic-sync \
  --split $SPLIT \
  --split-data-path $SPLIT_DATA_PATH \
  --veh-config-path $VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH \
  --device $CUDA_VISIBLE_DEVICES \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1
