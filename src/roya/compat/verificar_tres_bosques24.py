"""Verificación de ventanas, posterior, particiones e inferencia sin objetivo."""
import json
import joblib,numpy as np,pandas as pd
import tres_bosques_bayes24 as b

def run():
    rng=np.random.default_rng(42)
    # Dependencia conjunta XOR: los marginales no distinguen presencia.
    x=np.array([[i,j,k] for i in [.1,.9] for j in [.1,.9] for k in [.1,.9]])
    x=np.repeat(x,300,axis=0);y=(x>.5).sum(axis=1)%2==1
    model=b.JointBeta3(3,20).fit(x,y);p=model.predict(x)
    assert (p[y]>.9).all() and (p[~y]<.1).all()
    details=model.details(x[:20]);assert np.isfinite(details.probabilidad_roya).all()
    for decision in [np.zeros(len(x),bool),np.ones(len(x),bool)]:
        risk=b.outcomes(p,decision);assert np.allclose(risk.sum(axis=1),1)
        assert (risk>=0).all().all()
        assert np.allclose(risk.P_real_pos_pred_neg+risk.P_real_pos_pred_pos,p)
    d=b.g.assemble(b.ROOT/'geografia_roya/base_geografica_exploratoria.csv',b.ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(b.g.OUT/'comunidades_por_observacion.csv');bundle=joblib.load(b.OUT/'modelo.joblib')
    w=b.window24(d,bundle['quantity']['bank']);assert len(w)==len(d)
    assert len([v for v in w if '_termica_dmenos' in v])==48
    assert not any('humedad' in v for v in w)
    periods=b.g.periods(d);first=periods.min();mask=periods==first
    assert w.loc[mask,'germinacion_dias_observados'].eq(0).all()
    assert w.loc[mask,[f'germinacion_termica_dmenos{k}' for k in range(1,25)]].isna().all().all()
    # Segunda quincena sólo dispone de15d; nunca se rellenan9d con ceros.
    second=sorted(periods.unique())[1];assert w.loc[periods==second,'germinacion_dias_observados'].eq(15).all()
    # Fecha3: comparación directa con matrices día-1 y día-24, excluyendo hoy.
    i=d.index[periods==sorted(periods.unique())[2]][0];date=pd.Timestamp(sorted(periods.unique())[2][:7]+'-01')
    bank=bundle['quantity']['bank'];day=bank['dates'].get_loc(date);site=bank['xy'][b.g.site_ids(d.loc[[i]]).iloc[0]]
    for lag in [1,24]:assert w.loc[i,'germinacion_termica_dmenos'+str(lag)]==int(bank['arrays']['germinacion_respuesta_realizacion_1'][day-lag,site]>=.9)
    grouped=pd.DataFrame(dict(sitio=b.g.site_ids(d),conjunto=m.conjunto)).groupby('sitio').conjunto.nunique();assert grouped.max()==1
    oof=pd.read_csv(b.OUT/'entradas_OOF.csv',float_precision='round_trip');train=d[m.conjunto=='train'];assert np.array_equal(oof.fila_excel,train.fila_excel)
    groups=b.g.site_ids(train).to_numpy()
    for k in oof.fold.unique():assert not set(groups[oof.fold==k])&set(groups[oof.fold!=k])
    candidates=d[m.conjunto=='prueba'];ix=rng.choice(candidates.index,96,replace=False);sample=d.loc[ix].drop(columns=[b.g.TARGET]).reset_index(drop=True)
    prediction=b.predict(bundle,sample);saved=pd.read_csv(b.OUT/'predicciones.csv').set_index('fila_excel').loc[d.loc[ix,'fila_excel']]
    for col in ['probabilidad_roya','incidencia_lineal_multiple','incidencia_log1p_lineal','incidencia_predicha']:
        assert np.allclose(prediction[col],saved[col],rtol=1e-6,atol=1e-6),col
    assert np.array_equal(prediction.presencia_predicha,saved.presencia_predicha)
    assert not prediction.humedad_simulada_disponible.any()
    assert len(pd.read_csv(b.OUT/'ocho_combinaciones.csv'))==8
    # Priors, conteos y media posterior exactos.
    bayes=bundle['bayes'];ixs=bayes.indices(oof[['p_contexto','p_previos24','p_individual']].to_numpy())
    n=np.zeros_like(bayes.n_);k=np.zeros_like(bayes.k_)
    np.add.at(n,ixs,1);np.add.at(k,ixs,oof.presencia_real.astype(int))
    assert np.array_equal(n,bayes.n_) and np.array_equal(k,bayes.k_)
    assert np.allclose(bayes.a_,bayes.a0_+k) and np.allclose(bayes.b_,bayes.b0_+n-k)
    report=dict(estado='aprobado',pruebas=['XOR conjunto3D','normalización cuatro riesgos','48indicadores y ventana24sin hoy',
        'faltantes de primera/segunda quincena','partición y OOF por coordenadas','96inferencias sin incidencia',
        'dos regresiones reproducidas','humedad ausente explícita','ocho combinaciones','posterior Beta exacto'])
    (b.OUT/'verificacion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False));return report

if __name__=='__main__':run()
