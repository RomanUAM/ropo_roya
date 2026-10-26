"""Pruebas de fuga, códigos, enmascarado, selección e inferencia persistida."""
import json
import numpy as np
import pandas as pd
import joblib
import corrector_cuatro_estados as c
from sistema_integrado_roya import predict_profile

def run():
    assert c.truth_codes([False,True,True,False],[False,False,True,True]).tolist()==[0,-1,1,2]
    assert len(c.FOREST_FEATURES)==12
    assert all(v not in c.FOREST_FEATURES for v in ['Planta',c.g.TARGET,'Coordenada_X','Coordenada_Y','Altitud','mes_sen'])
    r=json.loads((c.OUT/'resultados.json').read_text());b=joblib.load(c.OUT/'modelo.joblib')
    assert all('codigo' not in v and 'incidencia' not in v.lower() and c.g.TARGET!=v for v in r['meta_features'])
    oof=pd.read_csv(c.OUT/'objetivos_OOF.csv')
    assert not oof.fila_excel.duplicated().any() and (oof.fold>=0).all()
    assert np.array_equal(oof.objetivo_error.to_numpy(),c.truth_codes(oof.presencia_real,oof.decision_OOF))
    assert set(oof.objetivo_error)==set(c.CODES)
    pred=pd.read_csv(c.OUT/'predicciones.csv');assert len(pred)==28872
    assert not pred.fila_excel.duplicated().any()
    for s,g in pred.groupby('conjunto'):
        a=c.g.stats(g[c.g.TARGET].to_numpy()>0,g.presencia_predicha.to_numpy())
        expected=next(v for v in r['comparacion'] if v['conjunto']==s and v['variante']=='corrector_recomendado')
        assert all(a[k]==expected[k] for k in ['TN','FP','FN','TP'])
        p=g[['prob_estado_'+str(k) for k in c.CODES]].to_numpy();assert np.allclose(p.sum(axis=1),1)
        assert (g.loc[g.bosque_positivo,['prob_estado_-1','prob_estado_0']].to_numpy()==0).all()
        assert (g.loc[~g.bosque_positivo,['prob_estado_1','prob_estado_2']].to_numpy()==0).all()
        assert np.array_equal(g.codigo_error_real.to_numpy(),c.truth_codes(g[c.g.TARGET]>0,g.bosque_positivo))
    # Selección reproducible sólo desde validación.
    rows=pd.read_csv(c.OUT/'politicas_validacion.csv').to_dict('records')
    choice=c.choose_policy(rows,r['seleccion']['baseline']);assert (choice is not None)==r['seleccion']['adoptar']
    frozen=json.loads((c.OUT/'seleccion_antes_de_prueba.json').read_text());assert frozen==r['seleccion']
    base=pd.read_csv(c.ROOT/'geografia_roya/base_geografica_exploratoria.csv')
    sample=pred.groupby(['Mes_registrado','Quincena','conjunto'],sort=True).head(2)
    raw=sample[['fila_excel']].merge(base,on='fila_excel',validate='one_to_one',sort=False)
    # Sin variable objetivo: ninguna etapa puede depender de la incidencia actual.
    raw=raw.drop(columns=c.g.TARGET)
    calculated=predict_profile('corrector_cuatro_estados',raw)
    assert np.array_equal(calculated.presencia_predicha,sample.presencia_predicha.to_numpy())
    assert np.array_equal(calculated.codigo_error_estimado,sample.codigo_error_predicho.to_numpy())
    assert np.allclose(calculated.incidencia_predicha,sample.incidencia_predicha.to_numpy())
    original=predict_profile('corrector_cuatro_estados_bosque',raw)
    assert np.array_equal(original.presencia_predicha,sample.bosque_positivo.to_numpy())
    # Memoria modifica el siguiente plan: no repetir búsqueda con misma huella.
    from orquestador_aprendizaje_roya import ModelingAgent,fingerprint,memory_load
    memory=memory_load();fp=fingerprint()
    assert memory['numero_informes_historicos']==len(memory['historial_documentado'])>=23
    assert len(memory['lecciones_operativas'])>=7
    for source in memory['historial_documentado']:
        import hashlib
        assert hashlib.sha256((c.ROOT/source['fuente']).read_bytes()).hexdigest()==source['sha256']
    assert ModelingAgent().propose(memory,fp)['accion']=='verificar_sin_reentrenar'
    changed=ModelingAgent().propose(memory,'datos_nuevos_no_usados')
    assert changed['accion']=='entrenar_y_comparar'
    assert 0<=min(changed['exponentes'])<=max(changed['exponentes'])<=1
    result=dict(estado='correcto',predicciones_reproducidas=len(sample),sin_objetivo_en_inferencia=True,
        estados_compatibles=True,seleccion_solo_validacion=True,memoria_cambia_siguiente_plan=True)
    (c.OUT/'verificacion.json').write_text(json.dumps(result,indent=2));print(result)
    return result

if __name__=='__main__':run()
