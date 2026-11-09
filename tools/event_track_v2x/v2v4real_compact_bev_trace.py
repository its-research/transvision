"""Lossless spatial cells needed by post-head bilinear sampling.

Full classification/regression heads and all-anchor scores remain mandatory.
This records original float32 BEV cells, without quantization or replacing the
original predictor. It certifies no network forward or unsampled BEV values.
"""
import numpy as np

from .v2v4real_native_numeric_reference import (
    decode_heads, stable_observed_top64)


def required_cells(boxes, shape, lidar_range):
    shape=np.asarray(shape)
    boxes=np.asarray(boxes,dtype=np.float64)
    bounds=np.asarray(lidar_range,dtype=np.float64)
    if (shape.shape!=(4,) or shape.dtype.kind not in 'iu' or shape[0]!=1 or np.any(shape<1)
            or boxes.ndim!=2 or boxes.shape[1]!=7 or not np.isfinite(boxes).all()
            or bounds.shape!=(6,) or not np.isfinite(bounds).all()
            or np.any(bounds[3:]-bounds[:3]<=0)):
        raise ValueError('invalid compact BEV sampling geometry')
    _,channels,height,width=map(int,shape)
    fractions=(boxes[:,:2]-bounds[:2])/(bounds[3:5]-bounds[:2])
    valid=((fractions>=0)&(fractions<=1)).all(1)
    cells=set()
    for i in np.flatnonzero(valid):
        x,y=fractions[i]*[width,height]-.5
        x0,y0=int(np.floor(x)),int(np.floor(y))
        for yy in (y0,y0+1):
            for xx in (x0,x0+1):
                if 0<=xx<width and 0<=yy<height: cells.add((yy,xx))
    return np.asarray(sorted(cells),dtype=np.int64).reshape(-1,2)


def compact_trace(trace,lidar_range):
    if set(trace)!={'psm','regression','anchors','bev','observed_anchor_scores'}:
        raise ValueError('original complete trace fields required')
    boxes,scores=decode_heads(trace['psm'],trace['regression'],trace['anchors'])
    selected=stable_observed_top64(trace['observed_anchor_scores'],scores)
    bev=np.asarray(trace['bev'])
    if bev.dtype!=np.float32 or bev.ndim!=4 or not np.isfinite(bev).all():
        raise ValueError('original finite float32 BEV required')
    shape=np.asarray(bev.shape,dtype=np.int64)
    cells=required_cells(boxes[selected],shape,lidar_range)
    values=bev[0,:,cells[:,0],cells[:,1]].copy()  # rows=cells, columns=channels
    result={k:v for k,v in trace.items() if k!='bev'}
    result.update(bev_shape=shape,bev_cell_indices=cells,bev_cell_values=values)
    return result


def compact_features(*, boxes, lidar_range, bev_shape, bev_cell_indices, bev_cell_values):
    boxes=np.asarray(boxes,dtype=np.float64)
    expected=required_cells(boxes,bev_shape,lidar_range)
    cells=np.asarray(bev_cell_indices);values=np.asarray(bev_cell_values)
    shape=np.asarray(bev_shape);_,channels,height,width=map(int,shape)
    if (cells.dtype!=np.int64 or cells.shape!=expected.shape
            or not np.array_equal(cells,expected) or values.dtype!=np.float32
            or values.shape!=(len(cells),channels) or not np.isfinite(values).all()):
        raise ValueError('exact full sampling cell inventory and float32 bytes required')
    bounds=np.asarray(lidar_range,dtype=np.float64)
    fractions=(boxes[:,:2]-bounds[:2])/(bounds[3:5]-bounds[:2])
    valid=((fractions>=0)&(fractions<=1)).all(1)
    lookup={tuple(cell):value.astype(np.float64) for cell,value in zip(cells,values)}
    sampled=np.zeros((len(boxes),channels),dtype=np.float64)
    for i in np.flatnonzero(valid):
        x,y=fractions[i]*[width,height]-.5
        x0,y0=int(np.floor(x)),int(np.floor(y))
        for yy in (y0,y0+1):
            for xx in (x0,x0+1):
                if 0<=xx<width and 0<=yy<height:
                    sampled[i]+=lookup[(yy,xx)]*(1-abs(x-xx))*(1-abs(y-yy))
    pooled=np.empty((len(boxes),128),dtype=np.float64)
    for k in range(128):
        lo,hi=k*channels//128,((k+1)*channels+127)//128
        pooled[:,k]=sampled[:,lo:hi].mean(1)
    norms=np.linalg.norm(pooled,axis=1);valid &= norms>1e-12
    result=np.zeros_like(pooled);result[valid]=pooled[valid]/norms[valid,None]
    return result,valid
