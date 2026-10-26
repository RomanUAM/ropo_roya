import json,joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from bosque_regresiones_gp import ROOT,OUT,predict
from modelo_roya_50_20_30 import assemble,inputs,splitter,TARGET
from programacion_genetica_roya import levels
from sistema_integrado_roya import predict_profile

def run():
    threadpool_limits(2);b=joblib.load(OUT/'modelo.joblib');r=json.loads((OUT/'resultados.json').read_text())
    source=joblib.load(ROOT/'evaluacion_bosque_bayes/modelo.joblib')
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);pd.testing.assert_frame_equal(assignment,b['particion_plantas'])
    f=inputs(d,b['simulation_bank']);positive=(split=='train')&(d[TARGET]>0)
    assert np.allclose(b['positive_pp'][0].mean_,f[positive].mean().to_numpy())
    all_p=[]
    for row in r['comparacion_prueba']:
        name=row['modelo'];p=predict(b,d[split=='prueba'],name)
        saved=pd.read_csv(OUT/('predicciones_'+name+'.csv'))
        assert np.allclose(p.incidencia_predicha,saved.incidencia_predicha)
        assert np.allclose(p.probabilidad_presencia,saved.probabilidad_presencia)
        assert np.isfinite(p.incidencia_predicha).all()
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
        assert np.array_equal(p.presencia_predicha,predict(b,d[split=='prueba']).presencia_predicha)
        if name.startswith('genetica_'):
            model=b['models'][name];assert levels(model.tree_)<=3 and model.max_levels_seen_<=3
            assert np.allclose(model.scaler_.mean_,b['positive_pp'].transform(f[positive]).mean(axis=0))
        print(name,'OK: 17,280 predicciones reproducidas.')
    raw=pd.read_csv(ROOT/'geografia_roya/base_geografica_exploratoria.csv').iloc[:40]
    for profile,name in [('bosque_regresiones_gp',b['selected_model']),('bosque_genetica3',b['selected_gp']),
                         ('bosque_regresion_red3','red_6_9_12'),('bosque_regresion_red4','red_6_9_18_36'),
                         ('bosque_regresion_red5','red_6_9_18_36_36')]:
        assert np.allclose(predict_profile(profile,raw).incidencia_predicha,predict(b,d.iloc[:40],name).incidencia_predicha)
    print('OK: todas las rutas integradas, mismas plantas, bosque y ceros comunes.')

if __name__=='__main__':run()
