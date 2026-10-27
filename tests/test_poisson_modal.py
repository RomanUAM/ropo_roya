import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from roya.poisson_modal import scenario,audit_split,train_epochs,month_basis

def test_noise_is_relative_reproducible_and_never_changes_zero():
    a=np.r_[0.,np.full(10000,12.)];b=scenario(a,42)
    assert b[0]==0 and np.all(b>=.95*a) and np.all(b<=1.05*a)
    assert abs(b[1:].mean()-12)<.01
    np.testing.assert_array_equal(b,scenario(a,42))
    with pytest.raises(ValueError):scenario([-1])

def test_split_requires_both_classes_and_disjoint_coordinates():
    groups=np.repeat(np.arange(10),2);d=pd.DataFrame({'site':groups,'target':np.tile([0,1],10)})
    m=pd.DataFrame({'conjunto':np.repeat(['train']*5+['validacion']*2+['prueba']*3,2)})
    q=SimpleNamespace(g=SimpleNamespace(site_ids=lambda x:x.site,TARGET='target'))
    report=audit_split(q,d,m);assert [x['sitios'] for x in report]==[5,2,3]
    bad=m.copy();bad.loc[0,'conjunto']='prueba'
    with pytest.raises(ValueError,match='coordenada'):audit_split(q,d,bad)
    bad=d.copy();bad.loc[m.conjunto.eq('prueba'),'target']=0
    with pytest.raises(ValueError,match='presencia o ausencia'):audit_split(q,bad,m)

def test_poisson_network_learns_nonnegative_mean_with_zeros():
    rng=np.random.default_rng(7);x=rng.normal(size=(500,2));mu=np.exp(.4*x[:,0]);y=rng.poisson(mu)
    model,snaps=train_epochs(x,y,[8],800,42,[40,800]);p=model.predict(x)
    assert np.isfinite(p).all() and (p>0).all() and set(snaps)=={40,800}
    assert np.mean((p-mu)**2)<np.mean((y.mean()-mu)**2)

def test_month_is_categorical_and_cyclic():
    d=pd.DataFrame({'Mes_registrado':['2025-12-01','2025-01-01'],'Quincena':['1','2']});b=month_basis(d)
    assert b.shape==(2,14) and np.all(b[:,:12].sum(axis=1)==1)
    assert b[0,11]==1 and b[1,0]==1
