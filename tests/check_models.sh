bash ./tools/dist_train.sh configs/coformer/coformer_veh_only.py 8
bash ./tools/dist_train.sh configs/coformer/coformer.py 8
bash ./tools/dist_train.sh configs/ffnet/config_basemodel_veh_only.py 8
bash ./tools/dist_train.sh configs/ffnet/config_basemodel_fusion.py 8
bash ./tools/dist_train.sh configs/ffnet/config_ffnet_fusion.py 8
