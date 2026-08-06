_REGISTERED = False


def register_resilient_v2x_modules() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from transvision.dataset import transforms  # noqa: F401
    from transvision.dataset.transforms import formating  # noqa: F401
    from transvision.dataset import v2x_dataset  # noqa: F401
    from transvision.evaluation.metrics import dair_v2x_metric  # noqa: F401
    from transvision.models import data_preprocessors, dense_heads, detectors, hooks, necks, voxel_encoders  # noqa: F401
    from transvision.models.data_preprocessors import data_preprocessor  # noqa: F401

    # The legacy registration smoke test intentionally exposes only the
    # TRANSFORMS registry. Import the end-to-end adapters only when their
    # owning MMDetection3D registries exist.
    from mmdet3d import registry as mmdet3d_registry

    if hasattr(mmdet3d_registry, "DATASETS") and hasattr(
        mmdet3d_registry, "DATA_SAMPLERS"
    ):
        from transvision.dataset import resilient_v2x_mmdet  # noqa: F401
    if hasattr(mmdet3d_registry, "METRICS"):
        from transvision.evaluation.metrics import resilient_v2x_metric  # noqa: F401

    _REGISTERED = True
