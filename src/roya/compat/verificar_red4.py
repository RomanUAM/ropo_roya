"""Comprueba arquitectura y predicciones guardadas de las dos variantes."""
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,inputs,splitter
from sistema_integrado_roya import ROOT,predict_profile

def run(variants=None,layers=(6,9,18,36)):
    variants=('red4_completas','red4_acp') if variants is None else variants
    threadpool_limits(2)
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d)
    for name in variants:
        folder=ROOT/('evaluacion_'+name);b=joblib.load(folder/'modelo.joblib')
        assert b['detector'].hidden_layer_sizes==layers
        assert b['group_classifier'][-1].hidden_layer_sizes==layers
        pd.testing.assert_frame_equal(assignment,b['particion_plantas'])
        f=inputs(d,b['simulation_bank'])
        assert np.allclose(b['presence_pp'][0].mean_,f[split=='train'].mean().to_numpy())
        assert np.allclose(b['positive_pp'][0].mean_,f[(split=='train')&(d['Incidencia de roya del Café']>0)].mean().to_numpy())
        assert set(b['quantity_selected'])==set(range(1,7))
        for candidates in b['quantity_models'].values():
            assert candidates['red_'+'_'.join(map(str,layers))][-1].hidden_layer_sizes==layers
        p=predict_profile(name,d[split=='prueba']);saved=pd.read_csv(folder/'predicciones_prueba.csv')
        assert np.allclose(p.incidencia_predicha,saved.incidencia_predicha)
        assert np.allclose(p.probabilidad_presencia,saved.probabilidad_presencia)
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
        print(name,'OK: arquitectura, transformadores y 17,280 predicciones.')

if __name__=='__main__':run()
