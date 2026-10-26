"""Verificación numérica de BBB, HR histórica e inferencia sin incidencia."""
import json
import joblib,numpy as np,pandas as pd
import roya_humedad_bnn as b

def gradient_check():
    p=np.array([[.1,.2,.1],[.8,.9,.7],[.2,.3,.1],[.8,.8,.9]])
    y=np.array([False,True,True,False]);model=b.VariationalStateNN().initialize(p)
    x=model.scale.transform(p);ref=b.initial_decision(p);t=b.targets(y,ref);rng=np.random.default_rng(5);eps=[rng.normal(size=v.shape) for v in model.mu]
    loss,grad=model.loss_grad(x,t,ref,eps,len(x));maximum=0.
    for value,derivative in zip(model.mu+model.rho,grad):
        for ix in np.ndindex(value.shape):
            original=value[ix];h=1e-5;value[ix]=original+h;hi=model.loss_grad(x,t,ref,eps,len(x))[0]
            value[ix]=original-h;lo=model.loss_grad(x,t,ref,eps,len(x))[0];value[ix]=original
            numeric=(hi-lo)/(2*h);maximum=max(maximum,abs(numeric-derivative[ix]));assert np.isclose(numeric,derivative[ix],atol=5e-5,rtol=5e-3)
    return maximum

def run():
    maximum=gradient_check();bundle=joblib.load(b.OUT/'modelo.joblib')
    d=b.g.assemble(b.ROOT/'geografia_roya/base_geografica_exploratoria.csv',b.ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(b.g.OUT/'comunidades_por_observacion.csv');w=b.features(d,bundle['quantity']['bank'],bundle['humidity_history'])
    first=b.reference_dates(d).min();mask=b.reference_dates(d)==first
    assert w.loc[mask,'HR_previa24_n_mediciones'].eq(0).all() and w.loc[mask,'HR_previa24_media'].isna().all()
    mid=(d.Humedad_relativa>=75)&(d.Humedad_relativa<80)
    assert mid.any() and w.loc[mid,'HR_real_actual_ge75'].eq(1).all() and w.loc[mid,'HR_real_actual_ge80'].eq(0).all()
    assert w.HR_previa24_n_mediciones.max()==1 # Frecuencia real quincenal, no24d inventados.
    # Covariables futuras no pueden afectar resúmenes de antecedentes.
    dates=b.reference_dates(d);date=sorted(dates.unique())[5];ix=d.index[dates==date][:8]
    changed=bundle['humidity_history'].copy();changed.loc[pd.to_datetime(changed.fecha)>=date,'HR']=0
    a=b.features(d.loc[ix].reset_index(drop=True),bundle['quantity']['bank'],bundle['humidity_history'])
    z=b.features(d.loc[ix].reset_index(drop=True),bundle['quantity']['bank'],changed)
    assert np.allclose(a,z,equal_nan=True)
    # Objetivos de cuatro estados reconstruidos sólo en OOF train.
    oof=pd.read_csv(b.OUT/'entradas_OOF.csv',float_precision='round_trip');train=d[m.conjunto=='train']
    assert np.array_equal(oof.fila_excel,train.fila_excel)
    p=oof[['p_contexto','p_previos24_HR','p_individual']].to_numpy();ref=b.initial_decision(p)
    assert np.array_equal(oof.codigo_real_referencia,b.c.truth_codes(oof.presencia_real,ref))
    assert set(oof.codigo_real_referencia)==set(b.CODES)
    groups=b.g.site_ids(train).to_numpy()
    for k in oof.fold.unique():assert not set(groups[oof.fold==k])&set(groups[oof.fold!=k])
    # Reproducibilidad sin columna de roya y máscara de estados incompatibles.
    rng=np.random.default_rng(42);ix=rng.choice(d.index[m.conjunto=='prueba'],96,replace=False)
    result=b.predict(bundle,d.loc[ix].drop(columns=[b.g.TARGET]).reset_index(drop=True))
    saved=pd.read_csv(b.OUT/'predicciones.csv').set_index('fila_excel').loc[d.loc[ix,'fila_excel']]
    for col in ['probabilidad_roya','posterior_inferior95','posterior_superior95','incidencia_lineal_multiple','incidencia_log1p_lineal','incidencia_predicha']:
        assert np.allclose(result[col],saved[col],atol=1e-6,rtol=1e-6),col
    assert np.array_equal(result.presencia_predicha,saved.presencia_predicha)
    columns=['P_FN_inicial','P_TN_inicial','P_TP_inicial','P_FP_inicial'];assert np.allclose(result[columns].sum(axis=1),1)
    negative=~result.referencia_mayoria
    assert np.allclose(result.loc[negative,['P_TP_inicial','P_FP_inicial']],0)
    assert np.allclose(result.loc[~negative,['P_FN_inicial','P_TN_inicial']],0)
    assert (result.desviacion_posterior>0).any()
    risk=['P_real_neg_pred_neg','P_real_pos_pred_neg','P_real_neg_pred_pos','P_real_pos_pred_pos']
    assert np.allclose(result[risk].sum(axis=1),1)
    r=dict(estado='aprobado',error_maximo_gradiente=maximum,
        pruebas=['Gradientes de todos los pesos y dispersiones BBB por diferencias finitas','HR75/80continua y umbrales',
        'Sin conversión quincenal a días húmedos','Sin covariables futuras en antecedentes','OOF y sitios separados',
        'Cuatro objetivos reales reconstruidos','96inferencias sin incidencia','Máscara y normalización de cuatro estados',
        'Dispersión posterior no nula','Dos regresiones reproducidas'])
    (b.OUT/'verificacion.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False));return r

if __name__=='__main__':run()
