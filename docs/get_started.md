# Get Started

## Prerequisites

- gcc==10
- CUDA==11.7
- python==3.8
- torch==2.0.1

```shell
pip install --upgrade git+https://github.com/klintan/pypcd.git
pip install torch torchvision torchaudio
pip install -U openmim
mim install mmengine 'mmcv>=2.0.0rc4' 'mmdet>=3.0.0'
# git clone https://github.com/open-mmlab/mmdetection3d.git
# cd mmdetection3d
# git checkout v1.2.0
# pip install -v -e .
mim install 'mmdet3d==1.2.0'
```

```shell
pip install -v -e .
```
