# transvision

## FFNet Training

Firstly, train the basemodel on `DAIR-V2X` without latency

```shell
# Single-gpu training
cd ${FFNET-VIC_repo}
# export PYTHONPATH=$PYTHONPATH:./
# CUDA_VISIBLE_DEVICES=$1 python tools/train.py ffnet_work_dir/config_basemodel.py
bash ./tools/dist_train.sh configs/ffnet/config_basemodel_car.py 8
```

Secondly, put the trained basemodel in a folder `ffnet_work_dir/pretrained-checkpoints`.

Thirdly, train `FFNET` on `DAIR-V2X` with latency

```shell
# Single-gpu training
cd ${FFNET-VIC_repo}
export PYTHONPATH=$PYTHONPATH:./
CUDA_VISIBLE_DEVICES=$1 python tools/train.py ffnet_work_dir/config_ffnet.py
```

## Reference

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) commit: cf92b8d9d91bd54bbdecc254550fcbc7c65b5dc7
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D) commit:e2b7b7421efceb59d51f65b43c0f05bf031c3d0a
- [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)
