# check=skip=FromPlatformFlagConstDisallowed
FROM --platform=linux/amd64 mambaorg/micromamba:2.8.1@sha256:79284aa2949ac9555eca7e975ceb4ecefefc5964a2d4fdd4e181ca8eaccf347e AS micromamba

FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:bd746eb3b9953805ebe644847a227e218b5da775f47007c69930569a75c9ad7d

ENV PATH=/opt/resilient-v2x/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    PYTHONNOUSERSITE=1 \
    RESILIENT_V2X_CONTAINER_IMAGE_DIGEST=sha256:bd746eb3b9953805ebe644847a227e218b5da775f47007c69930569a75c9ad7d

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY environments/resilient_v2x/bootstrap-linux-64.explicit.txt /tmp/bootstrap-linux-64.explicit.txt
RUN micromamba create --yes --prefix /opt/bootstrap --file /tmp/bootstrap-linux-64.explicit.txt \
    && micromamba clean --all --yes

COPY environments/resilient_v2x/environment-linux-64.lock.yml /tmp/environment-linux-64.lock.yml
RUN /opt/bootstrap/bin/conda-lock install --micromamba --prefix /opt/resilient-v2x /tmp/environment-linux-64.lock.yml

WORKDIR /workspace/transvision
COPY setup.py setup.cfg MANIFEST.in README.md /workspace/transvision/
COPY environments/resilient_v2x/constraints.txt /workspace/transvision/environments/resilient_v2x/constraints.txt
COPY transvision /workspace/transvision/transvision
COPY configs /workspace/transvision/configs
COPY tools /workspace/transvision/tools
COPY scripts/build_resilient_v2x_ops.sh /workspace/transvision/scripts/build_resilient_v2x_ops.sh
RUN /opt/resilient-v2x/bin/python -c "import sys, torch, torchvision, cv2, mmengine, mmcv, mmdet, mmdet3d; assert sys.version_info[:2] == (3, 10); assert torch.__version__ == '2.0.1+cu118'; assert torchvision.__version__ == '0.15.2+cu118'; assert torch.version.cuda == '11.8'; assert cv2.__version__ == '4.11.0'; assert mmengine.__version__ == '0.10.7'; assert mmcv.__version__ == '2.1.0'; assert mmdet.__version__ == '3.2.0'; assert mmdet3d.__version__ == '1.3.0'" \
    && /bin/bash scripts/build_resilient_v2x_ops.sh \
    && groupadd --gid 10001 resilient-v2x \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash resilient-v2x \
    && chown -R 10001:10001 /workspace/transvision

USER 10001:10001
RUN /opt/resilient-v2x/bin/python -c "import cv2; import mmcv._ext, transvision.models.voxel.voxel_layer, transvision.models.bev_pool.bev_pool_ext"

CMD ["/opt/resilient-v2x/bin/python", "tools/resilient_v2x/check_environment.py", "--mode", "controlled", "--require-cuda", "--require-custom-ops"]
