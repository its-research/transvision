"""Uncropped CoopTrack detector decoding for the paper candidate protocol.

Install before install_raw_detector_decode, which supplies the pre-summary
head tensors. This adapter alone does not prevent tracking-summary mutation.
No candidate selection or score calibration is performed here.
"""


DECODE_SOURCE = "raw-head-all-queries-no-roi-no-nms-v1"


def decode_queries(logits, normalized_boxes, denormalize, pc_range):
    import torch
    if (logits.ndim != 2 or logits.shape[1] != 3
            or normalized_boxes.shape != (logits.shape[0], 10)):
        raise ValueError("expected Qx3 logits and Qx10 normalized boxes")
    if not torch.isfinite(logits).all() or not torch.isfinite(normalized_boxes).all():
        raise ValueError("non-finite raw detector head")
    scores, labels = logits.sigmoid().max(dim=-1)
    boxes = denormalize(normalized_boxes, pc_range)
    if boxes.shape != (logits.shape[0], 9) or not torch.isfinite(boxes).all():
        raise ValueError("invalid decoded geometry")
    if (boxes[:, 3:6] <= 0).any():
        raise ValueError("nonpositive decoded box dimensions")
    return {"boxes": boxes, "scores": scores, "labels": labels,
            "query_indices": torch.arange(logits.shape[0], device=logits.device)}


def install_uncropped_query_decode(model, denormalize):
    """Replace only detector result decoding; training/tracking remain forbidden."""
    if (model.training or not model.train_det or model.is_cooperation
            or model.STReasoner.history_reasoning or model.STReasoner.future_reasoning):
        raise ValueError("requires eval-only single-frame detector")
    if getattr(model, "_raw_detector_decode_installed", False):
        raise ValueError("install uncropped decoder before the raw-head snapshot adapter")
    if getattr(model, "_uncropped_query_decode_installed", False):
        raise ValueError("uncropped decoder already installed")
    state = {"frames": 0, "last_query_count": None, "decode_source": DECODE_SOURCE}

    def decode(prediction, metadata):
        if len(metadata) != 1:
            raise ValueError("single-frame metadata required")
        values = decode_queries(prediction["all_cls_scores"][-1],
                                prediction["all_bbox_preds"][-1],
                                denormalize, model.pts_bbox_head.bbox_coder.pc_range)
        boxes = metadata[0]["box_type_3d"](values["boxes"], 9)
        state["frames"] += 1
        state["last_query_count"] = len(values["scores"])
        return {"boxes_3d_det": boxes.to("cpu"),
                "scores_3d_det": values["scores"].cpu(),
                "labels_3d_det": values["labels"].cpu(),
                "query_indices_det": values["query_indices"].cpu()}

    model._det_instances2results = decode
    model._uncropped_query_decode_installed = True
    return state
