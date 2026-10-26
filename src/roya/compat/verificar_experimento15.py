import json,joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from experimento_15dias import ROOT,OUT,PRIMARY,REGRESSORS,predict
from ventana15_roya import features
from modelo_roya_50_20_30 import assemble,splitter,TARGET
from verificar_ventana15 import run as verify_window
from sistema_integrado_roya import predict_profile

def run():
    threadpool_limits(2);verify_window();b=joblib.load(OUT/'modelo.joblib');r=json.loads((OUT/'resultados.json').read_text())
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);pd.testing.assert_frame_equal(assignment,b['particion_plantas'])
    f=features(d,b['bank']);assert f.shape==(57600,30)
    count=f[b['bank']['columns']].to_numpy();valid=count[np.isfinite(count)]
    assert ((valid>=0)&(valid<=15)&(valid==np.floor(valid))).all()
    assert np.allclose(b['preprocessor'][1].mean_,b['preprocessor'][0].transform(f[split=='train']).mean(axis=0))
    positive=(split=='train')&(d[TARGET]>0)
    assert np.allclose(b['positive_preprocessor'][1].mean_,b['positive_preprocessor'][0].transform(f[positive]).mean(axis=0))
    assert len(b['primary_models'])==6 and set(b['regressors'])==set(REGRESSORS)
    oof=pd.read_csv(OUT/'ejemplos_confirmadores_OOF.csv');assert len(oof)==28800
    assert (oof.groupby(['Municipio','Parcela','Planta']).fold.nunique()==1).all()
    selected=oof.positivo_detector_OOF
    assert set(oof.etiqueta_confirmador[selected])=={0,1}
    assert oof.etiqueta_confirmador[~selected].isna().all()
    for route in ['sin_confirmacion']+list(b['routes']):
        p=predict(b,d[split=='prueba'],route=route);stored=pd.read_csv(OUT/('predicciones_ruta_'+route+'.csv'))
        assert np.allclose(p.incidencia_predicha,stored.incidencia_predicha)
        assert not (p.presencia_predicha&~p.positivo_detector).any()
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
    for name in REGRESSORS:
        p=predict(b,d[split=='prueba'],regression=name);stored=pd.read_csv(OUT/('predicciones_regresion_'+name+'.csv'))
        assert np.allclose(p.incidencia_predicha,stored.incidencia_predicha)
    raw=pd.read_csv(ROOT/'geografia_roya/base_geografica_exploratoria.csv').iloc[:40]
    assert np.allclose(predict_profile('ventana15',raw).incidencia_predicha,predict(b,d.iloc[:40]).incidencia_predicha)
    print('OK: conteos15, seis detectores, confirmadores OOF, 13 regresores e inferencia integrada.')

if __name__=='__main__':run()
