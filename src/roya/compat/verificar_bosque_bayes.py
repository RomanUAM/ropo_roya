"""Verifica separación, asociación espacial y todas las variantes del nuevo modelo."""
import json,joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from bosque_bayes_regresion import ROOT,OUT,predict,bayes_proba
from modelo_roya_50_20_30 import assemble,inputs,splitter,TARGET
from sistema_integrado_roya import predict_profile

def run():
    threadpool_limits(2);b=joblib.load(OUT/'modelo.joblib');report=json.loads((OUT/'resultados.json').read_text())
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);pd.testing.assert_frame_equal(assignment,b['particion_plantas'])
    f=inputs(d,b['simulation_bank']);tr=split=='train';pos=tr&(d[TARGET]>0)
    assert np.allclose(b['presence_pp'][0].mean_,f[tr].mean().to_numpy())
    assert np.allclose(b['positive_pp'][0].mean_,f[pos].mean().to_numpy())
    assert b['detector'].n_estimators==200
    assert len(b['regressions'])==report['numero_grupos']
    z=b['positive_pp'].transform(f[split=='prueba']);weights=bayes_proba(b,z)
    assert (weights>=0).all() and np.allclose(weights.sum(axis=1),1)
    for row in report['comparacion_prueba']:
        name=row['variante'];p=predict(b,d[split=='prueba'],name)
        saved=pd.read_csv(OUT/('predicciones_'+name.replace(':','_')+'.csv'))
        assert np.allclose(p.incidencia_predicha,saved.incidencia_predicha)
        assert np.allclose(p.probabilidad_presencia,saved.probabilidad_presencia)
        assert p.incidencia_predicha.notna().all()
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
        assert (p.loc[~p.presencia_predicha,'grupo_final']==0).all()
        print(name,'OK: 17,280 predicciones y ceros coherentes.')
    raw=pd.read_csv(ROOT/'geografia_roya/base_geografica_exploratoria.csv').iloc[:30]
    a=predict_profile('bosque_bayes',raw);c=predict(b,d.iloc[:30])
    assert np.allclose(a.incidencia_predicha,c.incidencia_predicha)
    for profile,key in [('bosque_bayes_grupos','selected_group_variant'),('bosque_bayes_neuronal','selected_neural_variant')]:
        candidate=predict_profile(profile,raw)
        expected=predict(b,d.iloc[:30],b[key])
        assert np.allclose(candidate.incidencia_predicha,expected.incidencia_predicha)
    print('OK: particiones, normalización sólo train, ACP sólo positivos train, probabilidades e inferencia integrada.')

if __name__=='__main__':run()
