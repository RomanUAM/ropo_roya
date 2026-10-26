"""Verifica decisiones y evidencia sobre todas las predicciones guardadas."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    root=Path(__file__).resolve().parent
    comparison=pd.read_csv(root/'evaluacion_integrada/comparacion.csv')
    assert len(comparison)==9
    for name in comparison.variante:
        p=pd.read_csv(root/'evaluacion_integrada'/(name+'.csv'))
        assert len(p)==24000
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
        assert p.loc[p.incidencia_predicha.isna(),'presencia_predicha'].all()
    p=pd.read_csv(root/'evaluacion_integrada/arbitro_3_9_27.csv')
    threshold=json.loads((root/'evidencia/arbitro_neuronal.json').read_text())['umbral_arbitro']
    disagreement=p.grupo_red!=p.grupo_bosque
    expected=np.where(disagreement,p.probabilidad_arbitro>=threshold,p.grupo_red>0)
    assert np.array_equal(expected,p.presencia_predicha)
    group=np.where(expected,np.where(p.grupo_red>0,p.grupo_red,p.grupo_bosque),0)
    assert np.array_equal(group,p.grupo_final)
    assert disagreement.any()
    assert (disagreement & (p.grupo_red==0)).any()
    assert (disagreement & (p.grupo_red>0) & (p.grupo_bosque>0)).any()
    p=pd.read_csv(root/'evaluacion_integrada/doble_red_svm.csv')
    assert np.array_equal(p.intervino_svm,p.grupo_red_5x5!=p.grupo_red_3_9_27)
    agree=~p.intervino_svm
    assert np.array_equal(p.loc[agree,'grupo_final'],p.loc[agree,'grupo_red_5x5'])
    assert p.intervino_svm.any()
    p=pd.read_csv(root/'evaluacion_integrada/doble_red_arbitro_profundo.csv')
    assert np.array_equal(p.intervino_arbitro,p.grupo_red_5x5!=p.grupo_red_3_9_27)
    agree=~p.intervino_arbitro
    assert np.array_equal(p.loc[agree,'grupo_final'],p.loc[agree,'grupo_red_5x5'])
    print('Verificadas nueve variantes, 216,000 filas y las reglas de arbitraje.')

if __name__=='__main__':main()
