import numpy as np,pytest
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.optimize._numdiff import approx_derivative
from roya.modelos.fusion_acotada import FractionalFusion,TruncatedPIGFusion,pig_logmass,BoundedNN
from roya.acp_probabilidades import gated,validate_probabilities

def test_pig_mass_agrees_with_independent_mixture_integral():
    mu=3.;shape=5.;lp=pig_logmass([mu],shape)[0]
    def mass(k):
        def integrand(t):return np.exp(k*np.log(t)-t-gammaln(k+1)+.5*np.log(shape/(2*np.pi))-1.5*np.log(t)-shape*(t-mu)**2/(2*mu*mu*t))
        return quad(integrand,0,np.inf,epsabs=1e-11)[0]
    raw=np.array([mass(k) for k in range(41)]);raw/=raw.sum()
    assert np.allclose(np.exp(lp),raw,atol=1e-9,rtol=1e-6) and np.isclose(np.exp(lp).sum(),1)

def test_pig_gradient_and_noninteger_rejection():
    x=np.array([[1.,-.5,.2,.4],[1.,.1,-.3,.7],[1.,1.,.5,-.2]]);y=np.array([0,4,9]);model=TruncatedPIGFusion(.01);params=np.array([-1.,.1,.2,-.1,2.]);loss,grad=model.loss_grad(params,x,y)
    numeric=approx_derivative(lambda z:model.loss_grad(z,x,y)[0],params,method='3-point').ravel()
    assert np.allclose(grad,numeric,atol=1e-5,rtol=1e-4)
    with pytest.raises(ValueError,match='enteros'):model.fit(x[:,1:],np.array([.5,4,9]))

def test_gam_and_polynomials_learn_and_stay_bounded():
    x=np.random.default_rng(42).uniform(0,40,(200,3));y=40/(1+np.exp(-(x[:,0]-20)/8))
    for kind in ['gam','poly2','poly3']:
        model=FractionalFusion(kind,.0001,5).fit(x,y);p=model.predict(x)
        assert np.mean((p-y)**2)<np.var(y)*.15 and np.all((p>=0)&(p<=40))

def test_neural_output_is_bounded_and_learns():
    # Dataset pequeño:400actualizaciones, frente a miles en80epochs de la base.
    x=np.random.default_rng(42).normal(size=(200,3));y=40/(1+np.exp(-x[:,0]));model=BoundedNN((12,6),400,42).fit(x,y);p=model.predict(x)
    assert np.mean((p-y)**2)<np.var(y)*.7 and np.all((p>=0)&(p<=40))

def test_regression_gate_never_executes_negatives():
    class Bomb:
        def transform(self,*args):raise AssertionError('Transformó negativos')
        def predict(self,*args):raise AssertionError('Predijo negativos')
    import pandas as pd
    result=gated(dict(design=Bomb(),model=Bomb()),pd.DataFrame({'x':[1,2]}),np.zeros((2,3)),[False,False])
    assert np.array_equal(result,[0.,0.])
    with pytest.raises(ValueError):validate_probabilities([[.2,2,.3]])

def test_neural_pipeline_does_not_execute_negative_quantity_networks():
    from unittest.mock import patch
    import pandas as pd
    from roya.tres_redes import predict
    class Bomb:
        def transform(self,*a):raise AssertionError('Transformó negativo')
        def predict(self,*a):raise AssertionError('Predijo negativo')
    fake=dict(models={'GAM':dict(model=Bomb())},networks=[dict(preprocess=Bomb(),network=Bomb())]*3,classifier_family='LR',selection={'mejor':{'fusion':'GAM'}})
    d=pd.DataFrame({'fila_excel':[2,3]});frame=pd.DataFrame({'x':[0.,0.]});details=pd.DataFrame({'probabilidad_roya':[.01,.02]})
    with patch('roya.tres_redes.joblib.load',return_value=fake),patch('roya.acp_probabilidades.inference_inputs',return_value=(d,frame,np.zeros((2,3)),np.array([False,False]),details)):
        result=predict(d,'GAM')
    assert result.incidencia_predicha.eq(0).all() and result.cantidad_red_1.eq(0).all()

def test_neural_checkpoints_are_independent_and_reproducible():
    import pickle
    x=np.random.default_rng(6).normal(size=(80,3));y=40/(1+np.exp(-x[:,0]))
    net=BoundedNN((20,30,30,40),20,42).fit(x,y,checkpoints=[5,20])
    assert set(net.checkpoints_)=={5,20}
    first=net.checkpoints_[5];last=net.checkpoints_[20]
    assert not np.allclose(first.predict(x),last.predict(x))
    np.testing.assert_allclose(pickle.loads(pickle.dumps(first)).predict(x),first.predict(x))
    np.testing.assert_allclose(net.predict(x),last.predict(x))
