"""Verifica tipos, geografía, seis OLS, registro y puerta verdaderamente perezosa."""
import json,copy
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LinearRegression
import reentrenamiento_integral_roya as b

def run():
    d=b.assemble();m=pd.read_csv(b.OUT/'comunidades_por_fila.csv');split=m.conjunto.to_numpy();tr=split=='train';te=split=='prueba'
    for col in b.g.STAGES:assert set(d[col].unique())<={0,1}
    assert (d[b.g.STAGES].sum(axis=1)>1).any() and (d[b.g.STAGES].sum(axis=1)==0).any()
    north=d.Orientacion=='N';flat=d.Orientacion=='PLANO'
    assert np.allclose(d.loc[north,'orientacion_original_sen'],0) and np.allclose(d.loc[north,'orientacion_original_cos'],1)
    assert d.loc[flat,'terreno_plano_original'].eq(1).all() and np.allclose(d.loc[flat,['orientacion_original_sen','orientacion_original_cos']],0)
    for col in ['Coordenada_X','Coordenada_Y','Altitud','Pendiente_grad','orientacion_original_sen','orientacion_original_cos']:assert col in b.e.topo_columns()
    top=joblib.load(b.OUT/'topografia.joblib');assert len(top['train_keys'])==1197
    n=pd.DataFrame(dict(sitio=b.g.site_ids(d),conjunto=split)).groupby('sitio').conjunto.nunique();assert n.max()==1
    oofs=[pd.read_csv(b.OUT/('OOF_'+f+'.csv')) for f in b.FAMILIES]
    for o in oofs:
        assert np.array_equal(o.fila_excel,d.loc[tr,'fila_excel']) and np.array_equal(o.fold,oofs[0].fold)
        groups=b.g.site_ids(d[tr]).to_numpy()
        for k in o.fold.unique():assert not set(groups[o.fold==k])&set(groups[o.fold!=k])
    ref=b.load_reference();w=b.h.features(d,ref['quantity']['bank'],ref['humidity_history']);qf=b.e.quantity_frame(d,m,w);q=joblib.load(b.OUT/'regresiones.joblib')
    assert set(q['models'])==set(b.e.REGRESSIONS) and len(q['models'])==6
    assert all(isinstance(x,LinearRegression) for x in q['models'].values())
    assert not set(b.g.STAGES)&set(q['design'].num)
    assert set(b.g.STAGES)<=set(q['design'].bins)
    bits=q['design'].bits.transform(qf[q['design'].bins]);assert set(np.unique(bits))<={0,1}
    assert np.allclose(bits[:,[q['design'].bins.index(v) for v in b.g.STAGES]],d[b.g.STAGES])
    # Cambiar a falsa no puede ejecutar transform ni predict.
    class Bomb:
        def transform(self,*args):raise AssertionError('Se transformaron negativos.')
        def predict(self,*args):raise AssertionError('Se regresaron negativos.')
    fake=dict(design=Bomb(),models={name:Bomb() for name in b.e.REGRESSIONS},min_positive=1)
    for name in b.e.REGRESSIONS:assert np.allclose(b.e.predict_gated(fake,name,qf.head(10),np.zeros(10,bool)),0)
    # En puerta mixta, contador garantiza sólo número de positivos.
    class Count:
        def __init__(self,model):self.model=model;self.count=0
        def predict(self,x):self.count+=len(x);return self.model.predict(x)
    wrapper=Count(q['models']['lineal_multiple']);mixed=dict(q);mixed['models']=dict(q['models']);mixed['models']['lineal_multiple']=wrapper
    gate=np.array([True,False]*5);z=b.e.predict_gated(mixed,'lineal_multiple',qf.head(10),gate);assert wrapper.count==5 and np.allclose(z[~gate],0)
    # Predicción sólo sobre filas registradas, sin incidencia como entrada.
    from sistema_integrado_roya import predict_profile
    rng=np.random.default_rng(42);ix=rng.choice(d.index[te],48,replace=False);request=d.loc[ix].drop(columns=[b.g.TARGET]).reset_index(drop=True)
    comparison=pd.read_csv(b.OUT/'matrices_y_metricas_clasificacion.csv')
    for family in b.FAMILIES:
        pred=predict_profile('integral_'+family,request);saved=pd.read_csv(b.OUT/('predicciones_'+family+'.csv')).set_index('fila_excel').loc[request.fila_excel]
        assert np.array_equal(pred.presencia_predicha,saved.presencia_predicha)
        for name in b.e.REGRESSIONS:assert np.allclose(pred['nivel_'+name],saved['nivel_'+name],atol=1e-5,rtol=1e-5),name
        assert np.allclose(pred.incidencia_predicha,saved.incidencia_predicha,atol=1e-5,rtol=1e-5)
        assert (pred.loc[~pred.presencia_predicha,['nivel_'+name for name in b.e.REGRESSIONS]]==0).all().all()
        allpred=pd.read_csv(b.OUT/('predicciones_'+family+'.csv'));frame=allpred[allpred.conjunto=='prueba'];metrics=b.common.metrics(frame.incidencia_observada.to_numpy()>0,frame.presencia_predicha.to_numpy(bool));row=comparison[(comparison.familia==family)&(comparison.conjunto=='prueba')].iloc[0]
        for key in ['TN','FP','FN','TP','errores']:assert row[key]==metrics[key]
    try:b.registered_data(pd.DataFrame({'fila_excel':[int(d.fila_excel.max())+10000]}))
    except ValueError:pass
    else:raise AssertionError('Se aceptó fila externa.')
    changed=request.head(1).copy();changed['Brotacion']=1-changed.Brotacion
    try:b.registered_data(changed)
    except ValueError:pass
    else:raise AssertionError('Se aceptó registro alterado.')
    pca=q['design'].cont;positive_train=tr&(d[b.g.TARGET].to_numpy()>0)
    imputed=pca[0].transform(qf.loc[positive_train,q['design'].num]);assert np.allclose(pca[1].mean_,imputed.mean(axis=0))
    result=dict(estado='aprobado',pruebas=['Binarias multietiqueta sinACP ni escalado','Orientación circular y plano distinto deN','Topografía original y1197nodos train porparcela',
        'Partición y folds comunes porcoordenadas','Seis OLS y PCA sólo continuas de positivos train','Puerta negativa no llama transform ni predict',
        'Puerta mixta sólo regresa positivos','48predicciones porfamilia con seis magnitudes reproducidas','Tres matrices reproducidas','Rechazo de filas externas o alteradas'])
    b.write(b.OUT/'verificacion.json',result);print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':run()
