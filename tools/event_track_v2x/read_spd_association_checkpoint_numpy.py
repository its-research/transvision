"""Restricted read of hash-admitted CPU float32 tensor ZIP checkpoints, no Torch.

Only contiguous/strided tensors with the specific float-storage reconstruction
are admitted; no general pickle globals or code objects are allowed.
"""
from collections import OrderedDict
import io,pickle,zipfile
import numpy as np
from scipy.special import erf


def read_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();data=[n for n in names if n.endswith('/data.pkl')];assert len(data)==1;prefix=data[0][:-len('data.pkl')];assert archive.read(prefix+'byteorder')==b'little'
        class FloatStorage:pass
        def tensor(storage,offset,shape,strides,requires_grad,hooks,*extra):
            assert isinstance(storage,np.ndarray) and storage.dtype==np.dtype('<f4') and len(shape)==len(strides) and offset>=0 and all(s>=0 for s in shape) and all(s>=0 for s in strides)
            last=offset+sum((s-1)*stride for s,stride in zip(shape,strides) if s>0);assert last<len(storage) or not np.prod(shape)
            return np.ndarray(tuple(shape),dtype='<f4',buffer=storage,offset=offset*4,strides=tuple(s*4 for s in strides)).copy()
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                allowed={('collections','OrderedDict'):OrderedDict,('torch','FloatStorage'):FloatStorage,('torch._utils','_rebuild_tensor_v2'):tensor}
                if (module,name) not in allowed:raise ValueError('unrecognized checkpoint pickle global')
                return allowed[module,name]
            def persistent_load(self,value):
                assert len(value)==5 and value[0]=='storage' and value[1] is FloatStorage and value[3]=='cpu' and type(value[4]) is int and 0<=value[4]<=10000000;key=str(value[2]);assert key.isdecimal();raw=archive.read(prefix+'data/'+key);assert len(raw)==4*value[4];return np.frombuffer(raw,dtype='<f4')
        result=Reader(io.BytesIO(archive.read(data[0]))).load()
    assert set(result['model'])=={'encoder.0.weight','encoder.0.bias','encoder.1.weight','encoder.1.bias','encoder.4.weight','encoder.4.bias','pair.0.weight','pair.0.bias','pair.3.weight','pair.3.bias','left_dustbin.weight','left_dustbin.bias','right_dustbin.weight','right_dustbin.bias'}
    assert all(np.isfinite(v).all() for v in result['model'].values());return result


def forward(checkpoint,left,right):
    w={k:np.asarray(v,np.float64) for k,v in checkpoint['model'].items()}
    def linear(x,name):return x@w[name+'.weight'].T+w[name+'.bias']
    def gelu(x):return .5*x*(1+erf(x/np.sqrt(2)))
    def encode(x):
        x=linear(np.asarray(x,np.float64),'encoder.0');mean=x.mean(-1,keepdims=True);var=((x-mean)**2).mean(-1,keepdims=True);x=(x-mean)/np.sqrt(var+1e-5)*w['encoder.1.weight']+w['encoder.1.bias'];return gelu(linear(gelu(x),'encoder.4'))
    l,r=encode(left),encode(right);a=np.broadcast_to(l[:,None,:],(len(l),len(r),128));b=np.broadcast_to(r[None,:,:],a.shape);pair=linear(gelu(linear(np.concatenate([a,b,np.abs(a-b),a*b],-1),'pair.0')),'pair.3')[...,0];return pair,linear(l,'left_dustbin')[...,0],linear(r,'right_dustbin')[...,0]
