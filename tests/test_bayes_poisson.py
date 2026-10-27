import numpy as np
from roya.modelos.bayes_poisson import BayesianPoissonMLP

def test_all_layers_have_variational_parameters_and_reproducible_positive_posterior():
    x=np.random.default_rng(1).normal(size=(100,3));y=np.maximum(0,np.round(2+x[:,0]));model=BayesianPoissonMLP(layers=3,width=13,epochs=10).fit(x,y)
    assert len(model.mu_)==8 and len(model.rho_)==8
    assert all(np.isfinite(m).all() for m in model.mu_+model.rho_)
    p=model.posterior(x,samples=20);q=model.posterior(x,samples=20)
    np.testing.assert_array_equal(p['media'],q['media'])
    assert (p['media']>0).all() and np.all(p['p975']>=p['p025']) and np.any(p['p975']>p['p025'])

def test_probability_scaling_does_not_amplify_unseen_rare_classes():
    from roya.modelos.bayes_poisson import ProbabilityHistoryScaler
    x=np.zeros((10,7));x[:,0]=1;x[:,-4:]=np.arange(40).reshape(10,4)
    scaler=ProbabilityHistoryScaler().fit(x);future=x[:1].copy();future[0,:3]=[0.,0.,1.]
    z=scaler.transform(future);np.testing.assert_array_equal(z[0,:3],[0.,0.,1.]);assert np.isfinite(z).all()
