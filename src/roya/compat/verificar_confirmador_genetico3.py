import json,joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from genetica_binaria3 import GeneticBinary3
from programacion_genetica_roya import levels
from confirmador_genetico3 import ROOT,OUT,infer
from modelo_roya_50_20_30 import assemble,splitter
from sistema_integrado_roya import predict_profile

def run():
    threadpool_limits(2)
    # Caso conocido y reproducibilidad: clases separadas, protección y límite de niveles.
    x=np.linspace(-2,2,100).reshape(-1,1);y=(x[:,0]>0).astype(int)
    a=GeneticBinary3(population=32,generations=5,seed=42).fit(x,y)
    c=GeneticBinary3(population=32,generations=5,seed=42).fit(x,y)
    assert a.tree_==c.tree_ and levels(a.tree_)<=3 and a.max_levels_seen_<=3
    assert (a.predict(x)==y).mean()>=.9
    assert np.allclose(a.predict_proba(x).sum(axis=1),1)
    b=joblib.load(OUT/'modelo.joblib');r=json.loads((OUT/'resultados.json').read_text())
    for item in r['candidatos']:
        m=b['confirmers'][item['model_key']];assert levels(m.tree_)<=3 and m.max_levels_seen_<=3
        assert all(z['levels']<=3 for z in m.history_)
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv');s,assignment=splitter(d)
    pd.testing.assert_frame_equal(assignment,b['particion_plantas'])
    for route in ['genetica3_conservadora','genetica3_balanceada']:
        p=infer(b,d[s=='prueba'],route);saved=pd.read_csv(OUT/('predicciones_'+route+'.csv'))
        assert len(saved)==17280 and np.allclose(p.incidencia_predicha,saved.incidencia_predicha)
        mask=p.positivo_detector;assert p.P_roya_condicional[mask].between(0,1).all()
        assert np.allclose((p.P_roya_condicional+p.P_no_roya_condicional)[mask],1)
        assert p.confirmacion_genetica_0_no_1_si[~mask].isna().all()
        assert not (p.presencia_predicha&~mask).any()
    assert np.allclose(predict_profile('confirmador_genetico3',d.iloc[:30]).incidencia_predicha,infer(b,d.iloc[:30]).incidencia_predicha)
    print('OK: GP binaria conocida, reproducibilidad, máximo 3 niveles, 17,280 predicciones por política e integración.')

if __name__=='__main__':run()
