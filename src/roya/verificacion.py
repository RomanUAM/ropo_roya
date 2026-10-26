"""Auditoría independiente de protocolos, modelos congelados e inferencia."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.inferencia import predict
from roya.modelos.magnitud import gated,uniform_by_id,metrics
from roya.registro import write

def verify(out):
    threadpool_limits(2);out=Path(out);q=load();d,m,w,_=q.data();split=m.conjunto.to_numpy();selection=json.loads((out/'seleccion_antes_prueba.json').read_text());tr=split=='train';te=split=='prueba'
    rows=pd.read_csv(out/'validacion_cantidad.csv');best=rows[rows.variante.str.endswith('uniforme')].sort_values(['R2_esperado_uniforme','RMSE'],ascending=[False,True]).iloc[0]
    assert best.modelo==selection['mejor_nivel']['modelo'] and best.familia==selection['mejor_nivel']['familia'] and best.representacion==selection['mejor_nivel']['representacion']
    def fail(*args):raise AssertionError('Se ejecutó regresión para negativos')
    class Bomb:
        transform=fail
        predict=fail
    fake=dict(design=Bomb(),models={'lineal_multiple':dict(model=Bomb(),smearing=1,C=2)},noise=.021834988)
    assert np.allclose(gated(fake,'lineal_multiple',pd.DataFrame({'x':[1,2]}),[False,False],stochastic=True,ids=[2,3]),0.)
    ids=d.loc[np.random.default_rng(42).choice(d.index[te],48,replace=False),'fila_excel'].to_numpy()
    cm=pd.read_csv(out/'clasificacion.csv');comparisons=pd.read_csv(out/'comparacion_cantidad.csv')
    assert len(comparisons)==448
    for rep in ['completas','ACP']:
        quantity=joblib.load(out/rep/'cantidad.joblib');assert quantity['design'].representation==rep
        assert not set(q.g.STAGES)&set(quantity['design'].num)
        assert set(q.g.STAGES)<=set(quantity['design'].bins)
        metadata=pd.read_csv(out/rep/'ajustes_cantidad.csv');assert metadata.n_train.eq(11900).all() and metadata.n_fit.eq(9518).all() and metadata.n_calibracion.eq(2382).all()
        for family in ['RF','LR','GP','MIXTA']:
            saved=pd.read_csv(out/rep/('predicciones_'+family+'.csv'));subset=saved[saved.conjunto=='prueba'];truth=subset.incidencia_observada.to_numpy()>0;gate=subset.presencia_predicha.to_numpy(bool)
            row=cm[(cm.representacion==rep)&(cm.familia==family)&(cm.conjunto=='prueba')].iloc[0];actual=q.old.common.metrics(truth,gate)
            for key in ['TN','FP','FN','TP']:assert row[key]==actual[key]
            got=predict(pd.DataFrame({'fila_excel':ids}),rep,family,out=out);expected=saved.set_index('fila_excel').loc[ids]
            assert np.array_equal(got.presencia_predicha,expected.presencia_predicha)
            assert np.allclose(got.cantidad_central,expected.incidencia_predicha,atol=2e-4,rtol=1e-5),(rep,family)
            assert got.loc[~got.presencia_predicha,'incidencia_predicha'].eq(0).all()
            assert got.dia.astype(int).isin([1,16]).all() and got.fecha_aproximada.eq(1).all()
            newer=joblib.load(out/rep/(family+'.joblib'))
            if family!='MIXTA':
                for model in newer['models'][1:]:
                    cats=model[0].named_transformers_['calendario'].feature_names_in_;assert set(cats)=={'mes_calendario','dia_calendario'}
                    if rep=='ACP':assert model[0].named_transformers_['continuas'][-1].n_samples_==int(tr.sum())
        # Efecto esperado de ruido uniforme: R² no puede mejorar al mismo centro.
        sub=comparisons[comparisons.representacion==rep]
        noisy=sub[sub.variante.str.endswith('uniforme')]
        assert (noisy.R2_esperado_uniforme<=noisy.R2+1e-12).all()
    original=uniform_by_id(ids,42,.021834988);reversed_values=uniform_by_id(ids[::-1],42,.021834988)[::-1];assert np.array_equal(original,reversed_values) and np.all(np.abs(original)<=.021834988)
    got=predict(pd.DataFrame({'fila_excel':ids[:4]}),'ACP','RF',stochastic=True,seed=7,out=out);again=predict(pd.DataFrame({'fila_excel':ids[:4]}),'ACP','RF',stochastic=True,seed=7,out=out)
    assert np.array_equal(got.incidencia_predicha,again.incidencia_predicha)
    try:predict(pd.DataFrame({'fila_excel':[999999999]}),out=out)
    except ValueError:pass
    else:raise AssertionError('Se aceptó fila externa')
    result=dict(estado='aprobado',pruebas=['448comparaciones=2representaciones*4arquitecturas*7regresiones*4variantes*2conjuntos','Selección porR2esperado de validación reproducida','Ocho matrices reproducidas','48filas por ocho rutas reproducidas sin objetivo','Negativos no llaman transform ni predict','Fecha aproximada explícita y calendario en clasificadores','PCAe indicadores separados;PCAclassifier ajustado sólo train','11900positivos train;calibración2382positivos fold0,fit9518','Ruido porfila/semilla reproducible, sin selección de realización','Ruido de media cero no mejoraR2esperado','Rechazo de fila externa'])
    write(out/'verificacion.json',result);return result
