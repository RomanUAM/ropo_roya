import json,joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from confirmador_red6 import ROOT,OUT
from experimento_15dias import predict
from modelo_roya_50_20_30 import assemble,splitter,TARGET
from ventana15_roya import features
from filtro_logistico_condicional import meta_features
from sistema_integrado_roya import predict_profile

def run():
    threadpool_limits(2);b=joblib.load(OUT/'modelo.joblib')
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);pd.testing.assert_frame_equal(assignment,b['particion_plantas'])
    model=b['confirmers']['red6'];assert model[-1].hidden_layer_sizes==(10,5,10,5,10,5)
    oof=pd.read_csv(ROOT/'evaluacion_15dias/ejemplos_confirmadores_OOF.csv');selected=oof.positivo_detector_OOF.to_numpy()
    f=features(d[split=='train'].reset_index(drop=True),b['bank'])
    x=meta_features(f[selected],oof.prob_detector_OOF.to_numpy()[selected])
    assert np.allclose(model[1].mean_,model[0].transform(x).mean(axis=0))
    assert (oof.groupby(['Municipio','Parcela','Planta']).fold.nunique()==1).all()
    for route in ['sin_confirmacion','red6_conservadora','red6_balanceada']:
        p=predict(b,d[split=='prueba'],route=route);saved=pd.read_csv(OUT/('predicciones_'+route+'.csv'))
        assert len(saved)==17280 and np.allclose(p.incidencia_predicha,saved.incidencia_predicha)
        assert not (p.presencia_predicha&~p.positivo_detector).any()
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
        if route!='sin_confirmacion':assert p.loc[p.positivo_detector,'probabilidad_condicional_confirmador'].between(0,1).all()
    raw=pd.read_csv(ROOT/'geografia_roya/base_geografica_exploratoria.csv').iloc[:40]
    assert np.allclose(predict_profile('confirmador_red6',raw).incidencia_predicha,predict(b,d.iloc[:40]).incidencia_predicha)
    print('OK: seis capas, sólo meta-train OOF, tres rutas de 17,280 filas, probabilidades y sistema integrado.')

if __name__=='__main__':run()
