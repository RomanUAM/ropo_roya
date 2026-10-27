import numpy as np
from roya.comunidades_regresion import overlay,GroupedLoss,communities,align,link,nondominated

def test_overlay_uses_all_posteriors_not_sparse_missing_edges_as_zeros():
    p=[np.array([[.7,.3],[.2,.8]]),np.array([[.1,.9],[.5,.5]]),np.array([[.9,.1],[.8,.2]])]
    a=overlay(p);assert np.isclose(a[0,1],sum(x[0]@x[1] for x in p)/3)
    assert np.allclose(a,a.T)

def test_grouped_losses_equal_every_original_row_without_subsampling():
    x=np.array([[1,0],[1,0],[1,1],[1,1],[1,1]],float);y=np.array([0.,2.5,1.,3.,8.]);g=GroupedLoss(x,y);p=np.array([[.2,3.],[4.,5.]])
    r2,mae,j=g.evaluate(p)
    for k in range(2):
        pred=p[g.inverse,k];assert np.isclose(mae[k],np.mean(abs(y-pred)))
        assert np.isclose(r2[k],1-np.mean((y-pred)**2)/np.var(y))
        assert np.isclose(j[k],1-r2[k]+mae[k]/np.std(y))

def test_holdout_assignments_cannot_change_training_communities():
    train=np.array([True,True,True,True,False]);a=np.array([[1,.9,.01,.01,.2],[.9,1,.01,.01,.2],[.01,.01,1,.9,.8],[.01,.01,.9,1,.8],[.2,.2,.8,.8,1.]])
    l,_=communities(a,train);b=a.copy();b[4,:]=.99;b[:,4]=.99;ll,_=communities(b,train)
    assert np.array_equal(l[train],ll[train]);assert l[0]==l[1] and l[2]==l[3] and l[0]!=l[2]

def test_alignment_and_links():
    train=np.ones(4,bool);old=np.array([0,0,1,1]);current=np.array([1,1,0,0]);aligned,_=align(current,old,train,2);assert np.array_equal(aligned,old)
    for kind in ['logistica','gompertz','hill']:
        p=link(np.array([-100.,0.,100.]),kind);assert np.isfinite(p).all() and np.all((p>=0)&(p<=40)) and np.all(np.diff(p)>=0)
    assert np.array_equal(nondominated(np.array([.5,.4,.6]),np.array([2.,3.,4.])),[True,False,True])
