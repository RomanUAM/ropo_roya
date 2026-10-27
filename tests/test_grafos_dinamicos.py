import numpy as np
import pytest
from roya.grafos_dinamicos import pair_probability,adjacency,change,Regimes

def test_coregime_probability_matches_independent_enumeration():
    p=np.array([[.8,.2],[.3,.7],[1.,0.]])
    assert pair_probability(p,0,1)==pytest.approx(.8*.3+.2*.7)
    assert pair_probability(p,0,2)==pytest.approx(.8)
    assert pair_probability(p,0,1)==pair_probability(p,1,0)
    with pytest.raises(ValueError):pair_probability(p,1,1)

def test_sparse_graph_is_symmetric_and_does_not_connect_missing_nodes():
    z=np.array([[0.,0.],[.1,0.],[10.,10.],[np.nan,0.]])
    p=np.array([[.9,.1],[.8,.2],[.1,.9],[np.nan,np.nan]])
    a=adjacency(z,p,k=2,threshold=.3)
    assert (a-a.T).nnz==0 and np.all(a.diagonal()==0)
    assert a[0,1]==pytest.approx(.74) and a[0,2]==0 and a[3].nnz==0
    assert change(a,a)==pytest.approx(1.) and change(None,a) is None

def test_regimes_are_frozen_for_transform_and_missingness_remains_unknown():
    rng=np.random.default_rng(42);x=np.r_[rng.normal(-3,.2,(60,2)),rng.normal(3,.2,(60,2))]
    model=Regimes().fit(x,max_components=2);old=model.scaler.mean_.copy();z,p,v=model.transform([[100.,100.],[np.nan,0.]])
    np.testing.assert_array_equal(model.scaler.mean_,old)
    assert np.allclose(p[0].sum(),1) and np.isnan(p[1]).all() and not v[1]
