import os
import os.path as osp
import sys

from transvision.models.detection_models import *

# from transvision.dataset.loading import LoadPointsFromFile

SUPPROTED_MODELS = {
    'single_side': SingleSide,
    'late_fusion': LateFusion,
    'early_fusion': EarlyFusion,
    'veh_only': VehOnly,
    'inf_only': InfOnly,
    'feature_fusion': FeatureFusion,
    'feature_flow': FeatureFlow,
}
