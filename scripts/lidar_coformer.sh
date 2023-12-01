DATA='data/DAIR-V2X/cooperative-vehicle-infrastructure'
SPLIT=val
SPLIT_DATA_PATH="data/split_datas/cooperative-split-data.json"
OUTPUT="cache/vic-coformer-baseline"
VEHICLE_MODEL_PATH='work_dirs/mmdet3d_1.3.0/coformer/basemodel/epoch_40.pth'
VEHICLE_CONFIG_NAME='configs/coformer/coformer.py'
CUDA_VISIBLE_DEVICES=$1 \

python transvision/eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model coformer \
  --dataset vic-sync \
  --split $SPLIT \
  --split-data-path $SPLIT_DATA_PATH \
  --veh-config-path $VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH \
  --device $CUDA_VISIBLE_DEVICES \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1
