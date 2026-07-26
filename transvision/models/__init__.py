def get_supported_models() -> dict[str, type]:
    from transvision.models.detection_models import (
        CoFormer,
        EarlyFusion,
        FeatureFlow,
        FeatureFusion,
        InfOnly,
        LateFusion,
        SingleSide,
        VehOnly,
    )

    return {
        "single_side": SingleSide,
        "late_fusion": LateFusion,
        "early_fusion": EarlyFusion,
        "veh_only": VehOnly,
        "inf_only": InfOnly,
        "feature_fusion": FeatureFusion,
        "feature_flow": FeatureFlow,
        "coformer": CoFormer,
    }


def __getattr__(name: str) -> object:
    if name == "SUPPROTED_MODELS":
        return get_supported_models()
    raise AttributeError(name)
