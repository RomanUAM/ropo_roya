import numpy as np
import pandas as pd
import pytest
from roya.niveles_bayes import labels,InverseBayes,pad_prob
from roya.modelos.genetica_ordinal import GeneticOrdinal3

def test_labels_do_not_drop_zero_or_high_values():
    np.testing.assert_array_equal(labels([0,1,3,4,6,7,36]),[0,1,1,2,2,3,12])
    np.testing.assert_array_equal(labels([.2,3.1]),[1,2])
    with pytest.raises(ValueError):labels([-1])

def test_inverse_bayes_is_finite_smoothed_and_prior_when_evidence_disabled():
    rng=np.random.default_rng(42);y=np.array([0]*50+[1]*25+[2]*5);p=rng.dirichlet([1,1,1],size=(80,3));ctx=pd.DataFrame({'topo':['a']*50+['b']*30,'clima':['x','y']*40})
    model=InverseBayes().fit(p,y,ctx);unknown=pd.DataFrame({'topo':['nueva']*4,'clima':['nueva']*4})
    pro=model.predict_proba(p[:4],unknown)
    assert np.isfinite(pro).all() and np.all(pro>0) and np.allclose(pro.sum(axis=1),1)
    np.testing.assert_allclose(model.predict_proba(p[:4],unknown,0,0),np.tile(model.prior_,(4,1)))

def test_probability_padding_preserves_class_identity_when_fold_lacks_a_class():
    class Model:
        classes_=np.array([0,2])
        def predict_proba(self,x):return np.array([[.2,.8]])
    p=pad_prob(Model(),[[1]],3);np.testing.assert_array_equal(p,[[.2,0,.8]])

def test_gp_ordinal_has_bounded_depth_and_valid_probabilities():
    rng=np.random.default_rng(1);x=rng.normal(size=(100,2));y=np.digitize(x[:,0],[-.5,.5]);model=GeneticOrdinal3(population=24,generations=3).fit(x,y);p=model.predict_proba(x)
    assert model.levels_<=3 and p.shape==(100,3) and np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
