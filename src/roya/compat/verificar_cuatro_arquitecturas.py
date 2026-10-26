"""Verifica selección sin prueba, reproducción y puerta de los siete estimadores."""
import json
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
import cuatro_arquitecturas_roya as q

def run():
    threadpool_limits(2);d,m,w,ref=q.data();split=m.conjunto.to_numpy();tr=split=='train';te=split=='prueba';y=d[q.g.TARGET].to_numpy(float)
    selections=json.loads((q.OUT/'seleccion_entradas_antes_prueba.json').read_text())['entradas']
    quality=pd.read_csv(q.OUT/'calidad_nueve_probabilidades_validacion.csv')
    oof=pd.read_csv(q.OUT/'OOF_MIXTA.csv')
    for item in selections:
        family=item['familia'];best=quality[quality.familia==family].sort_values(['Brier','log_loss','indice']).iloc[0]
        assert best.vista==item['vista']
        original=pd.read_csv(q.old.OUT/('OOF_'+family+'.csv'))
        assert np.array_equal(oof.fold,original.fold) and np.array_equal(oof.fila_excel,original.fila_excel)
        assert np.allclose(oof['p_'+family],original['p_'+item['vista']])
    coords=q.g.site_ids(d[tr]).to_numpy()
    for k in oof.fold.unique():assert not set(coords[oof.fold==k])&set(coords[oof.fold!=k])
    cs=pd.read_csv(q.OUT/'candidatos_MIXTA.csv').to_dict('records');chosen=min(cs,key=lambda r:(r['FP']+r['FN'],r['Brier'],-r['exactitud_balanceada']))
    saved=json.loads((q.OUT/'seleccion_MIXTA_antes_prueba.json').read_text());assert chosen['epoch']==saved['epoch'] and np.isclose(chosen['umbral'],saved['umbral'])
    vals=pd.read_csv(q.OUT/'validacion_regresiones.csv');selected=json.loads((q.OUT/'seleccion_antes_prueba.json').read_text())
    assert len(vals)==28
    for f in q.FAMILIES:assert vals[vals.familia==f].sort_values(['MAE','RMSE']).iloc[0].regresion==selected['regresion_por_familia'][f]
    neural=joblib.load(q.OUT/'red_cantidad.joblib');quantity=joblib.load(q.old.OUT/'regresiones.joblib');frame=q.e.quantity_frame(d,m,w)
    x=quantity['design'].transform('lineal_multiple',frame.loc[tr&(y>0)].reset_index(drop=True))
    assert np.allclose(neural['scaler'].mean_,x.mean(axis=0)) and neural['seleccion']['n_train']==11900
    class Bomb:
        def transform(self,*a):raise AssertionError('Se transformaron negativos')
        def predict(self,*a):raise AssertionError('Se ejecutó red sobre negativos')
    assert np.allclose(q.neural_gated(dict(scaler=Bomb(),model=Bomb()),Bomb(),frame.head(5),np.zeros(5,bool)),0)
    rng=np.random.default_rng(42);ids=d.loc[rng.choice(d.index[te],60,replace=False),'fila_excel'].to_numpy();request=pd.DataFrame({'fila_excel':ids})
    from sistema_integrado_roya import predict_profile
    for f in q.FAMILIES:
        got=predict_profile('cuatro_'+f,request,regression='red_neuronal');saved=pd.read_csv(q.OUT/('predicciones_'+f+'.csv')).set_index('fila_excel').loc[ids]
        assert np.array_equal(got.presencia_predicha,saved.presencia_predicha)
        for method in q.METHODS:assert np.allclose(got['nivel_'+method],saved['nivel_'+method],atol=1e-5,rtol=1e-5),(f,method)
        assert np.allclose(got.incidencia_predicha,got.nivel_red_neuronal)
        assert (got.loc[~got.presencia_predicha,['nivel_'+s for s in q.METHODS]]==0).all().all()
        allpred=pd.read_csv(q.OUT/('predicciones_'+f+'.csv'));a=allpred[allpred.conjunto=='prueba'];metrics=q.old.common.metrics(a.incidencia_observada.to_numpy()>0,a.presencia_predicha.to_numpy(bool))
        cm=pd.read_csv(q.OUT/'clasificacion.csv');row=cm[(cm.familia==f)&(cm.conjunto=='prueba')].iloc[0]
        for name in ['TN','FP','FN','TP']:assert metrics[name]==row[name]
        if f!='MIXTA':
            old=pd.read_csv(q.old.OUT/('predicciones_'+f+'.csv'))
            for method in q.e.REGRESSIONS:assert np.allclose(old['nivel_'+method],allpred['nivel_'+method])
    c=pd.read_csv(q.OUT/'variables_correlacion_por_fila.csv');missing=w.ventana24_incompleta.astype(bool)
    assert c.loc[missing,'germinacion_dias_favorables_observados'].isna().all()
    # Validate24day interval independently for a fixed row.
    ix=np.flatnonzero(~missing.to_numpy())[7];day=ref['quantity']['bank']['dates'].get_loc(q.h.reference_dates(d).iloc[ix]);site=q.g.site_ids(d).map(ref['quantity']['bank']['xy']).iloc[ix]
    expected=ref['quantity']['bank']['means']['germinacion_respuesta_media'][day-24:day,int(site)].mean()
    assert np.isclose(c.iloc[ix].germinacion_respuesta_termica_media24,expected)
    try:predict_profile('cuatro_MIXTA',pd.DataFrame({'fila_excel':[999999999]}))
    except ValueError:pass
    else:raise AssertionError('Se aceptó fila externa')
    result=dict(estado='aprobado',pruebas=['Selección por Brier de validación reproducida','Mismos cincofolds OOF y coordenadas disjuntas',
        'Checkpoint/umbral y28selecciones por validación reproducidos','Red de cantidad ajustada sólo en11900positivos train',
        'Puerta neuronal negativa no transforma ni predice','60filas por arquitectura y siete cantidades reproducidas sin objetivo',
        'Cuatro matrices reproducidas','Seis regresiones y puertas previas inalteradas','Ventana estrictamente24dprevios y faltantes explícitos','Fuera de base rechazado'])
    q.old.write(q.OUT/'verificacion.json',result);print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':run()
