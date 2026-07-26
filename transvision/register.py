_REGISTERED = False


def register_resilient_v2x_modules() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from transvision.dataset import transforms  # noqa: F401
    from transvision.dataset.transforms import formating  # noqa: F401
    from transvision.dataset import v2x_dataset  # noqa: F401
    from transvision.evaluation.metrics import dair_v2x_metric  # noqa: F401
    from transvision.models import data_preprocessors, dense_heads, detectors, hooks, necks  # noqa: F401
    from transvision.models.data_preprocessors import data_preprocessor  # noqa: F401

    _REGISTERED = True
