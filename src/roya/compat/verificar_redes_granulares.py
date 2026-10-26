"""Verifica la regla matemática, separación geográfica e inferencia persistida."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from redes_granulares_roya import ROOT,OUT,SimilarityLayer,site_ids,stats
from sistema_integrado_roya import predict_profile

def run():
    model=SimilarityLayer();x=np.array([[0.,0.],[1.,0.],[1.,1.],[8.,8.],[9.,8.]])
    w=model.fit(x);d=cdist(model.x,model.x,'mahalanobis',VI=model.precision)
    expected=(d<model.mean[:,None])&(d<model.mean[None,:])&((1-d/model.dmax)>=.3)
    np.fill_diagonal(expected,False)
    assert np.array_equal(w>0,expected) and np.allclose(w,w.T)
    assert not np.diag(w).any() and ((w==0)|((w>=.3)&(w<=1))).all()
    # Añadir otros sitios a un lote no altera la asignación del primero.
    assert np.allclose(model.weights(np.array([[2.,1.]])),model.weights(np.array([[2.,1.],[99.,99.]]))[:1])
    frozen=model.precision.copy();model.weights(np.array([[99.,99.]]));assert np.array_equal(model.precision,frozen)
    equal=SimilarityLayer();we=equal.fit(np.zeros((5,3)));assert not we.any() # regla estricta, medias cero
    assignment=pd.read_csv(OUT/'particion_ubicaciones.csv');assert not assignment.sitio.duplicated().any()
    pred=pd.read_csv(OUT/'predicciones.csv');assert len(pred)==28872 and not pred.fila_excel.duplicated().any()
    base=pd.read_csv(ROOT/'geografia_roya/base_geografica_exploratoria.csv')
    report=json.loads((OUT/'resultados.json').read_text())
    for s in ['validacion','prueba']:
        p=pred[pred.conjunto==s];ys=p['Incidencia de roya del Café'].to_numpy()>0
        actual=stats(ys,p.presencia_predicha.to_numpy())
        expected=next(v for v in report['etapas'] if v['conjunto']==s and v['etapa']=='cascada_completa')
        assert all(actual[k]==expected[k] for k in ['TN','FP','FN','TP'])
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
    # Selección de regresión procede de validación, nunca de error de prueba.
    eligible=[r for r in report['regresiones'] if r['conjunto']=='validacion' and not r['advertencias']]
    assert report['regresion_elegida']==min(eligible,key=lambda r:r['MAE'])['modelo']
    # Cada red de referencia contiene exclusivamente coordenadas train.
    bundle=joblib.load(OUT/'modelo.joblib');train=set(assignment.loc[assignment.conjunto=='train','sitio'])
    for block in bundle['blocks'].values():assert set(block['train_ids'])==train
    # Una muestra de cada periodo reservado reproduce predicciones almacenadas.
    sample=pred.groupby(['Mes_registrado','Quincena','conjunto'],sort=True).head(2)
    raw=sample[['fila_excel']].merge(base,on='fila_excel',validate='one_to_one',sort=False)
    a=predict_profile('redes_granulares',raw)
    assert np.array_equal(a.presencia_predicha.to_numpy(),sample.presencia_predicha.to_numpy())
    assert np.allclose(a.incidencia_predicha.to_numpy(),sample.incidencia_predicha.to_numpy())
    forced=predict_profile('redes_granulares_filtro05',raw)
    expected=(sample.bosque_positivo.to_numpy()|sample.rescatado.to_numpy())&(sample.p_filtro_fenologia_condicional.to_numpy()>=.5)
    assert np.array_equal(forced.presencia_predicha.to_numpy(),expected)
    assert 'Fecha_muestreo_exacta' not in raw
    result=dict(estado='correcto',regla_arcos=True,sin_solapamiento_geografico=True,
        reproduccion_inferencia=len(sample),registros_validacion_prueba=len(pred),regresiones=7)
    (OUT/'verificacion.json').write_text(json.dumps(result,indent=2));print(result)
    return result

if __name__=='__main__':run()
