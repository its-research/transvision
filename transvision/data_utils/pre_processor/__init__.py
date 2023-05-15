# -*- coding: utf-8 -*-


from transvision.data_utils.pre_processor.base_preprocessor import BasePreprocessor
from transvision.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor

__all__ = {"BasePreprocessor": BasePreprocessor, "SpVoxelPreprocessor": SpVoxelPreprocessor}


def build_preprocessor(preprocess_cfg, train):
    process_method_name = preprocess_cfg["core_method"]
    error_message = f"{process_method_name} is not found. " f"Please add your processor file's name in transvision/" f"data_utils/processor/init.py"
    assert process_method_name in ["BasePreprocessor", "VoxelPreprocessor", "BevPreprocessor", "SpVoxelPreprocessor"], error_message

    processor = __all__[process_method_name](preprocess_params=preprocess_cfg, train=train)

    return processor
