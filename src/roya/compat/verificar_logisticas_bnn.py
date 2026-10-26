"""Pruebas de sustitución real de clasificadores, OOF y matrices comparables."""
import json
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
import comparar_logisticas_bnn as b

def run():
    bundle=joblib.load(b.OUT/'modelo.joblib');d=b.g.assemble(b.ROOT/'geografia_roya/base_geografica_exploratoria.csv',b.ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(b.g.OUT/'comunidades_por_observacion.csv');tr=m.conjunto=='train';te=m.conjunto=='prueba'
    assert len(bundle['models'])==3 and all(isinstance(v[-1],LogisticRegression) for v in bundle['models'])
    assert all(v[-1].n_iter_.max()<3000 for v in bundle['models'])
    w=b.h.features(d,bundle['quantity']['bank'],bundle['humidity_history'])
    assert list(w)==bundle['window_features']
    for model,x in zip(bundle['models'][1:],[w,d[b.c.FOREST_FEATURES]]):
        pre=model[0];imputed=pre[0].transform(x[tr]);assert np.allclose(pre[1].mean_,imputed.mean(axis=0))
    assert bundle['bnn'].mu[0].shape==(3,8) and bundle['bnn'].mu[2].shape==(8,4)
    oof=pd.read_csv(b.OUT/'entradas_OOF.csv',float_precision='round_trip');rf_oof=pd.read_csv(b.h.OUT/'entradas_OOF.csv')
    assert np.array_equal(oof.fila_excel,rf_oof.fila_excel) and np.array_equal(oof.fold,rf_oof.fold)
    groups=b.g.site_ids(d[tr]).to_numpy()
    for k in oof.fold.unique():assert not set(groups[oof.fold==k])&set(groups[oof.fold!=k])
    p=oof[['p_'+name for name in b.NAMES]].to_numpy();assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    ref=b.h.initial_decision(p);assert np.array_equal(oof.codigo_real_referencia,b.c.truth_codes(oof.presencia_real,ref))
    saved=pd.read_csv(b.OUT/'predicciones.csv',float_precision='round_trip');comparison=pd.read_csv(b.OUT/'comparacion.csv')
    assert np.array_equal(saved.fila_excel,d.fila_excel)
    for tag in ['validacion','prueba']:
        t=saved[saved.conjunto==tag];stats=b.metrics(t[b.g.TARGET].to_numpy()>0,t.presencia_predicha.to_numpy(bool));row=comparison[(comparison.conjunto==tag)&(comparison.modelo=='LR_BNN')].iloc[0]
        for key in ['TN','FP','FN','TP','errores']:assert row[key]==stats[key]
    # Inferencia nueva sin incidencia: modelo, posterior y dos regresiones.
    rng=np.random.default_rng(42);ix=rng.choice(d.index[te],96,replace=False);sample=d.loc[ix].drop(columns=[b.g.TARGET]).reset_index(drop=True)
    from sistema_integrado_roya import predict_profile
    pred=predict_profile('tres_logisticas_HR_red_bayesiana',sample);expected=saved.set_index('fila_excel').loc[d.loc[ix,'fila_excel']]
    for col in ['probabilidad_roya','posterior_inferior95','posterior_superior95','incidencia_lineal_multiple','incidencia_log1p_lineal','incidencia_predicha']:
        assert np.allclose(pred[col],expected[col],atol=1e-6,rtol=1e-6),col
    assert np.array_equal(pred.presencia_predicha,expected.presencia_predicha)
    assert np.allclose(pred[['P_FN_inicial','P_TN_inicial','P_TP_inicial','P_FP_inicial']].sum(axis=1),1)
    # Comparación directa contra exactamente la ruta RF guardada.
    rf=pd.read_csv(b.h.OUT/'predicciones.csv');assert np.array_equal(rf.fila_excel,saved.fila_excel)
    boot=pd.read_csv(b.OUT/'bootstrap_pareado_por_sitio.csv').set_index('metrica')
    test=comparison[(comparison.conjunto=='prueba')&comparison.modelo.isin(['RF_BNN','LR_BNN'])].set_index('modelo')
    for name in ['FP','FN']:assert boot.loc[name,'diferencia_LR_menos_RF']==test.loc['LR_BNN',name]-test.loc['RF_BNN',name]
    report=dict(estado='aprobado',pruebas=['Tres logísticas reales convergentes','Escalado ajustado sólo con train',
        'Mismas variables y red3–8–4','Misma partición/folds por coordenadas','Probabilidades OOF y cuatro estados',
        'Matrices reproducidas desde predicciones','96inferencias sin incidencia desde el catálogo integrado',
        'Dos regresiones e intervalos reproducidos','Comparación pareada con RF de las mismas filas'])
    (b.OUT/'verificacion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False));return report

if __name__=='__main__':run()
