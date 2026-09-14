"""Prediction-only PointPillar heads -> native V2 payload arrays.

The native OpenCOOD anchor order is x,y,z,h,w,l,yaw. Decoding and rotated suppression are explicit here; this is a protocol adaptation, not a claim that all native post-processing
(including its evaluation ROI) is identical.
"""
import numpy as np


def decode_heads(psm, regression, anchors):
    import torch
    anchors = torch.as_tensor(anchors, device=regression.device, dtype=regression.dtype)
    if (psm.ndim != 4 or psm.shape[0] != 1 or regression.shape != (1, 7 * psm.shape[1], *psm.shape[2:]) or anchors.shape != (*psm.shape[2:], psm.shape[1], 7)
            or not all(torch.isfinite(x).all() for x in (psm, regression, anchors)) or (anchors[..., 3:6] <= 0).any()):
        raise ValueError('invalid single-frame native heads/anchors')
    a = anchors.reshape(-1, 7)
    delta = regression.permute(0, 2, 3, 1).reshape(-1, 7)
    diagonal = torch.sqrt(a[:, 4]**2 + a[:, 5]**2)
    boxes = torch.stack((delta[:, 0] * diagonal + a[:, 0], delta[:, 1] * diagonal + a[:, 1], delta[:, 2] * a[:, 3] + a[:, 2], torch.exp(delta[:, 3]) * a[:, 3],
                         torch.exp(delta[:, 4]) * a[:, 4], torch.exp(delta[:, 5]) * a[:, 5], delta[:, 6] + a[:, 6]), 1)
    if not torch.isfinite(boxes).all():
        raise ValueError('nonfinite decoded anchors')
    scores = torch.sigmoid(psm.permute(0, 2, 3, 1)).reshape(-1)
    return boxes, scores


def rotated_nms(boxes, scores, *, threshold=.15, minimum_score=.05, max_candidates=10000):
    from shapely.geometry import Polygon
    boxes, scores = np.asarray(boxes, float), np.asarray(scores, float)
    if (boxes.shape != (len(scores), 7) or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or not 0 <= threshold <= 1 or not 0 <= minimum_score <= 1
            or type(max_candidates) is not int or max_candidates < 1 or np.any(boxes[:, 3:6] <= 0)):
        raise ValueError('invalid bounded rotated suppression')
    order = sorted(np.flatnonzero(scores >= minimum_score), key=lambda i: (-scores[i], int(i)))
    if len(order) > max_candidates:
        raise ValueError('NMS resource limit exceeded; no silent candidate truncation')
    polygons = {}
    for i in order:
        x, y, _, _, w, length, yaw = boxes[i]
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * [length / 2, w / 2]
        polygons[i] = Polygon(corners @ np.array([[c, s], [-s, c]]) + [x, y])
    kept = []
    for i in order:
        rejected = False
        for j in kept:
            intersection = polygons[i].intersection(polygons[j]).area
            if intersection / (polygons[i].area + polygons[j].area - intersection) > threshold:
                rejected = True
                break
        if not rejected:
            kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def bev_features(bev, boxes, lidar_range):
    """Bilinear center sampling followed by fixed channel pooling to 128 and
    L2."""
    import torch
    import torch.nn.functional as F
    if bev.ndim != 4 or bev.shape[0] != 1 or len(lidar_range) != 6 or not torch.isfinite(bev).all():
        raise ValueError('one finite BEV frame required')
    bounds = torch.as_tensor(lidar_range, device=bev.device, dtype=bev.dtype)
    if not torch.isfinite(bounds).all() or (bounds[3:] - bounds[:3] <= 0).any():
        raise ValueError('invalid LiDAR range')
    if len(boxes) == 0:
        return np.empty((0, 128)), np.empty((0, ), dtype=bool)
    centers = torch.as_tensor(boxes[:, :2], device=bev.device, dtype=bev.dtype)
    grid = 2 * (centers - bounds[:2]) / (bounds[3:5] - bounds[:2]) - 1
    valid = (grid.abs() <= 1).all(1)
    sampled = F.grid_sample(bev, grid[None, :, None, :], mode='bilinear', align_corners=False)[0, :, :, 0].T
    pooled = F.adaptive_avg_pool1d(sampled[:, None, :], 128)[:, 0]
    norms = torch.linalg.vector_norm(pooled, dim=1)
    valid &= norms > 1e-12
    output = torch.where(valid[:, None], pooled / norms.clamp_min(1e-12)[:, None], torch.zeros_like(pooled))
    return output.detach().cpu().double().numpy(), valid.cpu().numpy()


def native_arrays(psm, regression, anchors, bev, *, lidar_range, covariance_diagonal, calibration, nms_threshold=.15, max_candidates=10000):
    boxes, scores = decode_heads(psm, regression, anchors)
    boxes, scores = boxes.detach().cpu().double().numpy(), scores.detach().cpu().double().numpy()
    selected = rotated_nms(boxes, scores, threshold=nms_threshold, max_candidates=max_candidates)
    boxes, scores = boxes[selected], scores[selected]
    appearance, valid = bev_features(bev, boxes, lidar_range)
    variance = np.asarray(covariance_diagonal, float)
    if variance.shape != (9, ) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('explicit positive train-selected state variances required')
    n = len(boxes)
    states = np.zeros((n, 9), dtype=np.float64)
    states[:, :3] = boxes[:, :3]
    states[:, 3:6] = boxes[:, [4, 5, 3]]  # Legacy MMDet width,length,height.
    states[:, 6] = (-boxes[:, 6] - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    # Covariance is supplied in physical x,y,z,length,width,height,yaw,vx,vy order.
    variance = variance[[0, 1, 2, 4, 3, 5, 6, 7, 8]]
    return dict(
        states=states,
        raw_scores=scores,
        scores=calibration.apply(scores),
        covariances=np.repeat(np.diag(variance)[None], n, 0),
        class_indices=np.zeros(n, dtype=np.int64),
        appearance=appearance,
        appearance_valid=valid)


class FrozenPointPillar:
    """Bind an instantiated native network/preprocessor to real weight bytes.

    Caller loads the pinned native source/environment. No model-selection or GT loader is accepted. Hook captures exactly the BEV tensor used by cls_head.
    """

    def __init__(self, model, preprocessor, anchors, *, checkpoint, checkpoint_sha256, device='cpu'):
        import torch

        from .detection_cache_v2 import sha_file
        if sha_file(checkpoint) != checkpoint_sha256:
            raise ValueError('PointPillar checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values()):
            raise ValueError('expected finite native state_dict')
        model.load_state_dict(state, strict=True)
        self.model, self.preprocessor, self.anchors = model.to(device).eval().requires_grad_(False), preprocessor, anchors
        self.device, self.checkpoint, self.checkpoint_sha256 = device, checkpoint, checkpoint_sha256
        from .recoverable_identity import model_digest
        self.model_sha256 = model_digest(self.model)

    def predict(self, points, **postprocess):
        import torch

        from .detection_cache_v2 import sha_file
        from .recoverable_identity import model_digest
        if (sha_file(self.checkpoint) != self.checkpoint_sha256 or model_digest(self.model) != self.model_sha256 or any(m.training for m in self.model.modules())
                or any(p.requires_grad for p in self.model.parameters())):
            raise ValueError('frozen detector changed')
        cloud = np.asarray(points, np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 4 or not np.isfinite(cloud).all():
            raise ValueError('finite XYZ-intensity point cloud required')
        processed = self.preprocessor.collate_batch([self.preprocessor.preprocess(cloud)])
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in processed.items()}
        captured = []
        hook = self.model.cls_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            with torch.inference_mode():
                outputs = self.model({'processed_lidar': batch})
                if len(captured) != 1:
                    raise ValueError('expected one native classifier BEV input')
                return native_arrays(outputs['psm'], outputs['rm'], self.anchors, captured[0], **postprocess)
        finally:
            hook.remove()
