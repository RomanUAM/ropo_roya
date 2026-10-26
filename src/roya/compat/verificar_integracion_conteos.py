"""Prueba inferencia integrada, transformación sólo en train y evidencia guardada."""
import joblib,numpy as np,pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,inputs,splitter,predict,TARGET
from sistema_integrado_roya import predict_profile,ROOT

def run():
    threadpool_limits(2)
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    s,assignment=splitter(d)
    for name in ('conteos_completas','conteos_acp','base_completas'):
        folder=ROOT/('evaluacion_'+name)
        b=joblib.load(folder/'modelo.joblib');f=inputs(d,b.get('simulation_bank'))
        pd.testing.assert_frame_equal(assignment,b['particion_plantas'])
        assert np.allclose(b['presence_pp'][0].mean_,f[s=='train'].mean().to_numpy())
        assert np.allclose(b['positive_pp'][0].mean_,f[(s=='train')&(d[TARGET]>0)].mean().to_numpy())
        for pp in (b['presence_pp'],b['positive_pp']):
            if name=='conteos_acp':assert isinstance(pp[-1],PCA)
            else:assert not any(isinstance(step,PCA) for _,step in pp.steps)
        assert set(b['quantity_selected'])==set(range(1,7))
        result=predict_profile(name,d[s=='prueba'])
        stored=pd.read_csv(folder/'predicciones_prueba.csv')
        assert np.allclose(result.incidencia_predicha,stored.incidencia_predicha)
        assert np.allclose(result.probabilidad_presencia,stored.probabilidad_presencia)
        assert (result.loc[~result.presencia_predicha,'incidencia_predicha']==0).all()
        raw=pd.read_csv(ROOT/'geografia_roya/base_geografica_exploratoria.csv').iloc[:12]
        result_raw=predict_profile(name,raw)
        result_full=predict(b,d.iloc[:12])
        assert np.allclose(result_raw.incidencia_predicha,result_full.incidencia_predicha)
        print(name,': OK; 17,280 predicciones reproducidas; train/positivos e inferencia sin relieve precargado.')

if __name__=='__main__':run()
