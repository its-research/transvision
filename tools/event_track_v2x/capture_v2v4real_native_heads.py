"""Read-only hooks for a separately audited native post-head numeric candidate.

No model, checkpoint, point reader, GT or task is created here. The caller must
bind the original frame/source/checkpoint before using this capture primitive.
It preserves predict's returned object and never substitutes an inference core.
"""
import hashlib
from pathlib import Path
import numpy as np

CORE_SHA='5138ba83aa381b722828265b4dfb4934bd53cc533685fd004db862b3158cd48b'


def capture_predict(detector, core, points, **postprocess):
    import torch
    path=Path(core.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CORE_SHA:
        raise ValueError('unchanged original inference core required')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError('TF32 must be disabled before numeric capture')
    heads,features=[],[]
    def feature_hook(module,args):
        if len(args)!=1 or not isinstance(args[0],torch.Tensor):
            raise ValueError('one classifier BEV tensor required')
        features.append(args[0].detach().clone())
        # Returning None leaves the original input unchanged.
    def head_hook(module,args,result):
        if (not isinstance(result,dict) or not all(k in result for k in ('psm','rm'))
                or not all(isinstance(result[k],torch.Tensor) for k in ('psm','rm'))):
            raise ValueError('original native classification/regression heads required')
        heads.append((result['psm'].detach().clone(),result['rm'].detach().clone()))
        # Returning None leaves the original output unchanged.
    handles=[]
    try:
        handles.append(detector.model.cls_head.register_forward_pre_hook(feature_hook))
        handles.append(detector.model.register_forward_hook(head_hook))
        result=detector.predict(points,**postprocess)
        if len(heads)!=1 or len(features)!=1:
            raise ValueError('exactly one original forward required per frame')
        psm,regression=heads[0];bev=features[0]
        if any(x.dtype!=torch.float32 or not torch.isfinite(x).all() for x in (psm,regression,bev)):
            raise ValueError('finite original float32 heads/BEV required')
        # Scores are evaluated by the unchanged original core on the actual
        # device, retaining its float32 ties for exact protocol-order auditing.
        with torch.inference_mode():
            _,scores=core.decode_heads(psm,regression,detector.anchors)
        values=dict(psm=psm,regression=regression,bev=bev,observed_anchor_scores=scores)
        arrays={k:np.ascontiguousarray(x.detach().cpu().numpy()).copy() for k,x in values.items()}
        arrays['anchors']=np.ascontiguousarray(np.asarray(detector.anchors)).copy()
        for x in arrays.values():x.flags.writeable=False
        if hashlib.sha256(path.read_bytes()).hexdigest()!=CORE_SHA:
            raise ValueError('original core changed during capture')
        return result,arrays
    finally:
        for handle in reversed(handles):handle.remove()
