"""Comprueba particiones, reproducción y decisiones del modelo de dos partes."""
from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from modelo_roya_50_20_30 import predict,assemble,inputs,splitter

def main():
    root=Path(__file__).resolve().parent
    b=joblib.load(root/'modelos/modelo_50_20_30.joblib')
    r=json.loads((root/'evaluacion_50_20_30/resultados.json').read_text())
    a=pd.read_csv(root/'evaluacion_50_20_30/particion_plantas.csv')
    keys=['Municipio','Parcela','Planta'];sets={}
    for name in ['train','validacion','prueba']:
        sets[name]=set(a.loc[a.conjunto==name,keys].itertuples(index=False,name=None))
    assert not sets['train']&sets['validacion']
    assert not sets['train']&sets['prueba']
    assert not sets['validacion']&sets['prueba']
    assert [len(sets[k]) for k in ['train','validacion','prueba']]==[1200,480,720]
    assert not any(c in r['features'] for c in ['Planta','Incidencia de roya del Café','Municipio','Parcela'])
    assert b['clusters'].n_clusters==6
    assert len(b['quantity_selected'])==6
    assert all(g['positivos_validacion']>0 for g in r['grupos'])
    d=assemble(root/'geografia_roya/base_geografica_exploratoria.csv',root/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,_=splitter(d);test=d[split=='prueba'];p=predict(b,test)
    saved=pd.read_csv(root/'evaluacion_50_20_30/predicciones_prueba.csv')
    assert len(p)==17280
    assert np.allclose(p.incidencia_predicha,saved.incidencia_predicha,equal_nan=True)
    assert np.array_equal(p.grupo_final,saved.grupo_final)
    assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
    assert p.loc[p.presencia_predicha,'grupo_final'].between(1,6).all()
    assert p.incidencia_predicha.notna().all()
    raw=test.drop(columns=[c for c in test.columns if c.startswith('Dist_') or c.endswith('MDE_contexto_m') or c.startswith('Pendiente_MDE') or c.startswith('Posicion_relativa_MDE') or c.startswith('Orientacion_ladera_MDE')])
    q=predict(b,raw)
    assert np.allclose(p.incidencia_predicha,q.incidencia_predicha)
    print('Verificadas 2,400 plantas disjuntas, seis grupos y 17,280 predicciones reproducidas con y sin columnas geográficas explícitas.')

if __name__=='__main__':main()
