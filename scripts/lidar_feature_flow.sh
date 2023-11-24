export PYTHONPATH=$PYTHONPATH

DELAY_K=$2
DATA='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/ffnet'
VAL_DATA_PATH='data/DAIR-V2X/cooperative-vehicle-infrastructure/mmdet3d_1.2.0_training/ffnet/flow_data_jsons/flow_data_info_val_'${DELAY_K}'.json'
OUTPUT="./cache/vic-feature-flow"
VEHICLE_MODEL_PATH='work_dirs/mmdet3d_1.2.0/ffnet-vic3d/flow/fusion/epoch_10.pth'
VEHICLE_CONFIG_NAME='./configs/ffnet/config_ffnet_fusion.py'

CUDA_VISIBLE_DEVICES=$1

python transvision/eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model feature_flow \
  --test-mode $3 \
  --dataset vic-sync \
  --val-data-path $VAL_DATA_PATH \
  --veh-config-path $VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH \
  --device $CUDA_VISIBLE_DEVICES \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1
