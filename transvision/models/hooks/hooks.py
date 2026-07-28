from mmdet3d.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class GradChecker(Hook):

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        for key, val in runner.model.named_parameters():
            if val.grad is None and val.requires_grad:
                runner.logger.warning(
                    "%s did not receive a gradient in train iteration %d",
                    key,
                    batch_idx,
                )
