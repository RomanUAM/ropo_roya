"""Posterior conjunto, OOF, riesgos de error e inferencia sin incidencia real."""
import json
import joblib
import numpy as np
import pandas as pd
import dos_bosques_consenso_bayes as b
from sistema_integrado_roya import predict_profile

def run():
    # Patrón XOR: dependencia conjunta que no se reduce a dos marginales.
    x=np.tile([[.1,.1],[.1,.9],[.9,.1],[.9,.9]],(40,1));y=np.tile([0,1,1,0],40)
    toy=b.JointBetaConsensus(bins=2,strength=20).fit(x,y);p=toy.predict(np.array([[.1,.1],[.1,.9],[.9,.1],[.9,.9]]))
    assert p[1]>p[0] and p[2]>p[3]
    assert np.allclose(p,[(0+10)/(40+20),(40+10)/(40+20),(40+10)/(40+20),(0+10)/(40+20)])
    r=json.loads((b.OUT/'resultados.json').read_text());model=joblib.load(b.OUT/'modelo.joblib')
    assert r['global_features']==b.GLOBAL and len(r['individual_features'])==12
    oof=pd.read_csv(b.OUT/'entradas_consenso_OOF.csv');old=pd.read_csv(b.c.OUT/'objetivos_OOF.csv')
    assert len(oof)==28728 and np.array_equal(oof.fila_excel,old.fila_excel)
    assert np.allclose(oof.p_individual,old.probabilidad_OOF) and np.array_equal(oof.fold,old.fold)
    assert pd.DataFrame(dict(sitio=b.c.g.site_ids(oof),fold=oof.fold)).groupby('sitio').fold.nunique().max()==1
    bayes=model['bayes_models'][model['selected_bayes']]
    assert bayes.n_.sum()==len(oof) and bayes.k_.sum()==oof.presencia_real.sum()
    assert np.allclose(bayes.a_,bayes.a0_+bayes.k_) and np.allclose(bayes.b_,bayes.b0_+bayes.n_-bayes.k_)
    result=pd.read_csv(b.OUT/'predicciones.csv');assert len(result)==28872 and not result.fila_excel.duplicated().any()
    assert result.probabilidad_roya.between(0,1).all()
    assert (result.limite_inferior95<=result.probabilidad_roya).all() and (result.limite_superior95>=result.probabilidad_roya).all()
    for scope in ['global','individual','consenso']:
        assert np.allclose(result['prob_correcto_'+scope]+result['prob_falso_negativo_'+scope]+result['prob_falso_positivo_'+scope],1)
    for tag,t in result.groupby('conjunto'):
        for name,col in [('bosque_global','presencia_global'),('bosque_individual','presencia_individual'),('consenso_balanceado','presencia_consenso_balanceado'),('recomendado','presencia_predicha')]:
            a=b.c.g.stats(t[b.c.g.TARGET].to_numpy()>0,t[col].to_numpy());expected=next(v for v in r['comparacion'] if v['conjunto']==tag and v['modelo']==name)
            assert all(a[k]==expected[k] for k in ['TN','FP','FN','TP'])
    assert r['seleccion']['modelo']==min(r['resoluciones_validacion'],key=lambda v:v['Brier'])['modelo']
    assert json.loads((b.OUT/'seleccion_antes_prueba.json').read_text())==r['seleccion']
    sample=result.groupby(['Mes_registrado','Quincena','conjunto'],sort=True).head(2)
    raw=pd.read_csv(b.ROOT/'geografia_roya/base_geografica_exploratoria.csv');data=sample[['fila_excel']].merge(raw,on='fila_excel',validate='one_to_one',sort=False).drop(columns=b.c.g.TARGET)
    predicted=predict_profile('dos_bosques_bayes_balanceado',data)
    assert np.array_equal(predicted.presencia_predicha,sample.presencia_consenso_balanceado.to_numpy())
    assert np.allclose(predicted.probabilidad_consenso,sample.probabilidad_roya.to_numpy())
    assert np.allclose(predicted.incidencia_predicha,sample.incidencia_consenso_balanceado.to_numpy())
    risk=np.where(predicted.presencia_predicha,1-predicted.probabilidad_consenso,predicted.probabilidad_consenso)
    assert np.allclose(risk,predicted.probabilidad_error_estimado)
    assert set(model['quantity']['regressors'])==set(b.SIX)
    info=dict(estado='correcto',predicciones_reproducidas=len(sample),dependencia_conjunta_verificada=True,posterior_beta_verificado=True,
        sin_incidencia_actual_en_entradas=True,OOF_por_coordenadas=True,riesgos_error_normalizados=True)
    (b.OUT/'verificacion.json').write_text(json.dumps(info,indent=2));print(info);return info

if __name__=='__main__':run()
