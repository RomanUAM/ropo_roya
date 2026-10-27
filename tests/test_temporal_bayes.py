import numpy as np
from sklearn.linear_model import LogisticRegression
from roya.temporal_bayes import fit_expert

def test_compression_matches_expanded_logistic_objective():
    rng=np.random.default_rng(2);x=rng.normal(size=(80,3));y=(x[:,0]>0).astype(int);ix=np.r_[np.arange(80),np.repeat(np.arange(20),4)]
    a=LogisticRegression(class_weight='balanced',solver='newton-cholesky',tol=1e-9,max_iter=1000).fit(x[ix],y[ix]);b=fit_expert(LogisticRegression(class_weight='balanced',solver='newton-cholesky',tol=1e-9,max_iter=1000),x[ix],y[ix],'LR')
    np.testing.assert_allclose(a.predict_proba(x),b.predict_proba(x),atol=1e-7)

def test_conditional_pipeline_uses_only_past_values_of_same_series():
    import pandas as pd
    from types import SimpleNamespace
    from unittest.mock import patch
    from roya.cascada_positivos import inference_frames
    d=pd.DataFrame({'Municipio':['a']*4,'Parcela':['p']*4,'Coordenada_X':[1.]*4,'Coordenada_Y':[2.]*4,'fila_excel':[1,2,3,4],'y':[0,2,3,4],'date':pd.date_range('2025-01-01',periods=4,freq='15D'),'x':[10,20,30,40]})
    q=SimpleNamespace(g=SimpleNamespace(TARGET='y'),h=SimpleNamespace(reference_dates=lambda z:z.date))
    def fake_frames(q,z,m,w):return {name:z[['x']].copy() for name in ['topografia','excel','simulacion']}
    with patch('roya.cascada_positivos.frames',side_effect=fake_frames):
        meta,a=inference_frames(q,d,None,None);changed=d.copy();changed.loc[2,'y']=999;_,b=inference_frames(q,changed,None,None)
    assert a['excel'].iloc[0].x==20 and a['excel'].iloc[0].roya_t_1==2 and a['excel'].iloc[0].roya_t_2==0
    pd.testing.assert_series_equal(a['excel'].iloc[0],b['excel'].iloc[0]);assert b['excel'].iloc[1].roya_t_1==999
