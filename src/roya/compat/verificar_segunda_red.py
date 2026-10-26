"""OOF anidado, ventanas previas, ausencia de fuga e inferencia de la cadena."""
import json
import joblib
import numpy as np
import pandas as pd
import segunda_red_roya as s
from sistema_integrado_roya import predict_profile

def run():
    r=json.loads((s.OUT/'resultados.json').read_text());b=joblib.load(s.OUT/'modelo.joblib')
    assert set(b['quantity']['regressors'])==set(s.REGRESSIONS) and len(s.REGRESSIONS)==6
    assert 'genetica_3_niveles' not in b['quantity']['regressors']
    assert len(r['feature_names'])==44 and s.c.g.TARGET not in r['feature_names']
    assert not any('real' in k or 'objetivo' in k for k in r['feature_names'])
    assert all(name in r['feature_names'] for name in ['Coordenada_X','Coordenada_Y','Altitud']+s.LAGS)
    tr=pd.read_csv(s.OUT/'entradas_red2_OOF.csv');stored=pd.read_csv(s.c.OUT/'objetivos_OOF.csv')
    assert len(tr)==28728 and not tr.fila_excel.duplicated().any()
    assert np.array_equal(tr.fila_excel,stored.fila_excel) and np.allclose(tr.probabilidad_bosque,stored.probabilidad_OOF)
    assert np.array_equal(tr.objetivo_error,s.c.truth_codes(tr.presencia_real,tr.decision_red1_OOF))
    sites=s.c.g.site_ids(tr);outer=pd.DataFrame(dict(sitio=sites,fold=tr.fold)).drop_duplicates()
    assert outer.groupby('sitio').fold.nunique().max()==1
    inner=pd.read_csv(s.OUT/'particiones_interiores.csv')
    for k in range(5):
        a=inner[inner.outer==k];held=set(outer.loc[outer.fold==k,'sitio']);fit=set(outer.loc[outer.fold!=k,'sitio'])
        assert not set(a.sitio)&held and set(a.sitio)==fit
        assert a.groupby('sitio').inner.nunique().max()==1 and set(a.inner)=={0,1,2}
    incomplete=tr.ventana15_incompleta==1
    assert incomplete.any() and tr.loc[incomplete,s.LAGS].isna().all().all()
    assert tr.loc[~incomplete,s.LAGS].isin([0.,1.]).all().all()
    # Construcción física:15 fechas previas exactas, sin hoy ni circularidad.
    raw=pd.read_csv(s.ROOT/'geografia_roya/base_geografica_exploratoria.csv')
    one=raw[(pd.to_datetime(raw.Mes_registrado).dt.month==9)&raw.Quincena.str.endswith('2')].iloc[[0]].reset_index(drop=True)
    bank=b['quantity']['bank'];f=s.previous15(one,bank);site=bank['xy'][s.c.g.site_ids(one).iloc[0]];day=bank['dates'].get_loc(pd.Timestamp('2024-09-16'))
    for stage in ['germinacion','apresorio']:
        expected=(bank['arrays'][stage+'_respuesta_realizacion_1'][day-15:day,site]>=.9).astype(float)
        cols=[f'cond_{stage}_dia_m{lag:02d}' for lag in range(15,0,-1)]
        assert np.array_equal(f[cols].iloc[0].to_numpy(),expected)
        assert f['alguna_cond_'+stage+'_15d'].iloc[0]==float(expected.any())
    for net in b['second_models'].values():assert net['pre'][-1].n_samples_seen_==len(tr)
    preds=pd.read_csv(s.OUT/'predicciones.csv');assert len(preds)==28872 and not preds.fila_excel.duplicated().any()
    for tag,t in preds.groupby('conjunto'):
        metrics=s.c.g.stats(t[s.c.g.TARGET].to_numpy()>0,t.presencia_predicha.to_numpy())
        expected=next(v for v in r['comparacion'] if v['conjunto']==tag and v['etapa']=='bosque_red1_red2')
        assert all(metrics[k]==expected[k] for k in ['TN','FP','FN','TP'])
        probs=t[['prob_red2_estado_'+str(k) for k in s.c.CODES]].to_numpy();assert np.allclose(probs.sum(axis=1),1)
        assert (t.loc[t.presencia_red1,['prob_red2_estado_-1','prob_red2_estado_0']].to_numpy()==0).all()
        assert (t.loc[~t.presencia_red1,['prob_red2_estado_1','prob_red2_estado_2']].to_numpy()==0).all()
        assert (t.loc[~t.presencia_predicha,'incidencia_predicha']==0).all()
    prior=pd.read_csv(s.c.OUT/'predicciones.csv').set_index('fila_excel')
    assert np.array_equal(preds.presencia_red1,prior.loc[preds.fila_excel,'presencia_predicha'].to_numpy())
    policy=json.loads((s.OUT/'seleccion_antes_prueba.json').read_text());assert policy==r['seleccion']
    selected=s.c.choose_policy(pd.read_csv(s.OUT/'politicas_validacion.csv').to_dict('records'),policy['baseline'])
    assert (selected is not None)==policy['adoptar']
    assert r['regresion_elegida']==min([v for v in r['regresiones'] if v['conjunto']=='validacion'],key=lambda v:v['MAE'])['modelo']
    sample=preds.groupby(['Mes_registrado','Quincena','conjunto'],sort=True).head(2)
    input_data=sample[['fila_excel']].merge(raw,on='fila_excel',validate='one_to_one',sort=False).drop(columns=s.c.g.TARGET)
    calculated=predict_profile('segunda_red_final',input_data)
    assert np.array_equal(calculated.presencia_predicha,sample.presencia_predicha.to_numpy())
    assert np.allclose(calculated.incidencia_predicha,sample.incidencia_predicha.to_numpy())
    first=predict_profile('segunda_red_control_red1',input_data)
    assert np.array_equal(first.presencia_predicha,sample.presencia_red1.to_numpy())
    from orquestador_segunda_red import plan
    from orquestador_aprendizaje_roya import memory_load
    assert plan(memory_load())['accion']=='verificar_sin_reentrenar'
    result=dict(estado='correcto',OOF_exterior=5,OOF_interior=3,entradas_red2=44,regresiones=6,
        predicciones_reproducidas=len(sample),sin_incidencia_actual_en_entradas=True,ventana15_excluye_hoy=True,particiones_sin_solapamiento=True)
    (s.OUT/'verificacion.json').write_text(json.dumps(result,indent=2));print(result)
    return result

if __name__=='__main__':run()
