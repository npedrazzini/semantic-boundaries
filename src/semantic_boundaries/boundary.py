import numpy as np
from scipy.stats import norm

def boundary(P,grid=50,density=0.40,box_offset=0.1,tightness="auto"):
    P=np.asarray(P,dtype=float)
    if P.ndim!=2 or P.shape[1]!=2:
        raise ValueError("P must be an (n,2) array of 2D coordinates")
    if tightness=="auto":
        q1,q3=np.percentile(P[:,0],[25,75])
        h=(q3-q1)/1.34
        tightness=4*1.06*min(np.sqrt(np.var(P[:,0])),h)*len(P)**(-1/5)
    x=P[:,0]
    y=P[:,1]
    gx=np.linspace(x.min(),x.max(),grid)
    gy=np.linspace(y.min(),y.max(),grid)
    h=np.repeat(tightness,2)/4
    ax=np.subtract.outer(gx,x)/h[0]
    ay=np.subtract.outer(gy,y)/h[1]
    matrax=np.reshape(norm.pdf(ax),(-1,len(x)),order="F")
    matray=np.reshape(norm.pdf(ay),(-1,len(x)),order="F")
    z=np.dot(matrax,matray.T)/(len(x)*h[0]*h[1])
    zero=np.where(z<density)
    zeroX=gx[zero[1]]
    zeroY=gy[zero[0]]
    rX=np.ptp(x)*box_offset
    rY=np.ptp(y)*box_offset
    pXmin=x.min()-rX
    pXmax=x.max()+rX
    pYmin=y.min()-rY
    pYmax=y.max()+rY
    borderX=np.concatenate([gx,gx,np.repeat(pXmin,len(gy)),np.repeat(pXmax,len(gy)),[pXmin,pXmin,pXmax,pXmax]])
    borderY=np.concatenate([np.repeat(pYmin,len(gx)),np.repeat(pYmax,len(gx)),gy,gy,[pYmin,pYmax,pYmin,pYmax]])
    extra=np.column_stack([np.concatenate([zeroX,borderX]),np.concatenate([zeroY,borderY])])
    x_with_boundary=np.vstack([P,extra])
    h0=np.zeros(len(extra))
    x1=np.linspace(x_with_boundary[:,0].min()-0.07,x_with_boundary[:,0].max()+0.07,100)
    y1=np.linspace(x_with_boundary[:,1].min()-0.07,x_with_boundary[:,1].max()+0.07,100)
    xgrid,ygrid=np.meshgrid(x1,y1)
    return x_with_boundary,x1,y1,xgrid,ygrid,h0