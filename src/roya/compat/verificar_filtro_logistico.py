import json,joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from filtro_logistico_condicional import ROOT,OUT,predict,meta_features
from modelo_roya_50_20_30 import assemble,inputs,splitter,TARGET
from sistema_integrado_roya import predict_profile

def run():
    threadpool_limits(2);b=joblib.load(OUT/'modelo.joblib');r=json.loads((OUT/'resultados.json').read_text())
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);pd.testing.assert_frame_equal(assignment,b['base']['particion_plantas'])
    oof=pd.read_csv(OUT/'predicciones_train_OOF.csv');tr=d[split=='train'].reset_index(drop=True)
    assert len(oof)==len(tr)==28800
    assert np.array_equal(oof.presencia_real.to_numpy(),tr[TARGET].to_numpy()>0)
    units=oof.groupby(['Municipio','Parcela','Planta']).fold.nunique()
    assert (units==1).all() and set(oof.fold)==set(range(5))
    selected=oof.positivo_RF_OOF.to_numpy();labels=oof.etiqueta_meta[selected].to_numpy()
    assert set(labels)=={0.,1.} and oof.etiqueta_meta[~selected].isna().all()
    f=inputs(tr,b['base']['simulation_bank']);mf=meta_features(f[selected],oof.probabilidad_RF_OOF.to_numpy()[selected])
    for model in b['logistic_models'].values():
        assert np.allclose(model[0].mean_,mf.mean().to_numpy())
    for variant in ['referencia','conservadora','balanceada']:
        p=predict(b,d[split=='prueba'],variant)
        stored=pd.read_csv(OUT/('predicciones_'+variant+'.csv'))
        assert np.allclose(p.incidencia_predicha,stored.incidencia_predicha)
        assert not (p.presencia_predicha&~p.positivo_bosque).any()
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
        q=p.probabilidad_condicional_verdadero_positivo
        assert q[~p.positivo_bosque].isna().all()
        assert q[p.positivo_bosque].between(0,1).all()
        print(variant,'OK: sólo predichos positivos, probabilidad válida y 17,280 resultados reproducidos.')
    raw=pd.read_csv(ROOT/'geografia_roya/base_geografica_exploratoria.csv').iloc[:40]
    for profile,variant in [('filtro_logistico',b['selected_variant']),('filtro_logistico_conservador','conservadora'),('filtro_logistico_balanceado','balanceada')]:
        assert np.allclose(predict_profile(profile,raw).incidencia_predicha,predict(b,d.iloc[:40],variant).incidencia_predicha)
    print('OK: etiquetas FP=0/TP=1, plantas OOF separadas, escalado sólo meta-train e integración.')

if __name__=='__main__':run()
