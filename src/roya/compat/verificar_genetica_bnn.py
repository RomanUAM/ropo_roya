"""GP real, protocolo idéntico en datos/selección y comparación reproducible."""
import json,copy
import joblib,numpy as np,pandas as pd
from sklearn.metrics import log_loss
from programacion_genetica_roya import levels,size,evaluate
from scipy.special import expit
from genetica_probabilistica3 import GeneticProbability3
import comparar_genetica_bnn as b

def run():
    rng=np.random.default_rng(42);x=rng.normal(size=(240,3));y=(x[:,0]>0).astype(int)
    a=GeneticProbability3(population=32,generations=4).fit(x,y);z=GeneticProbability3(population=32,generations=4).fit(x,y)
    assert a.expression_==z.expression_ and a.max_levels_seen_<=3
    assert log_loss(y,a.predict_proba(x)[:,1])<.65
    light=joblib.load(b.OUT/'modelo.joblib');bundle=b.resolve_bundle(light)
    bad=copy.copy(light);bad['reference_sha256']='0'*64
    try:b.resolve_bundle(bad)
    except ValueError:pass
    else:raise AssertionError('Se aceptó referencia con hash incorrecto.')
    d=b.g.assemble(b.ROOT/'geografia_roya/base_geografica_exploratoria.csv',b.ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv');m=pd.read_csv(b.g.OUT/'comunidades_por_observacion.csv');tr=m.conjunto=='train';te=m.conjunto=='prueba'
    m=b.b.extra_transform(d,m,bundle['extra_graphs']);w=b.h.features(d,bundle['quantity']['bank'],bundle['humidity_history'])
    inputs=[m[b.b.CONTEXT],w,d[b.c.FOREST_FEATURES]];protocol=light['protocolo_comun']
    assert [len(b.g.site_ids(d[m.conjunto==v]).unique()) for v in ['train','validacion','prueba']]==[1100,440,660]
    for name,f in zip(b.NAMES,inputs):assert list(f)==protocol['entradas'][name]
    # Todos usan las mismas variables: se inspecciona el primer transformador.
    rf=b.reference_bundle();lr=joblib.load(b.common.OUT/'modelo.joblib')
    for j,f in enumerate(inputs):
        assert isinstance(light['models'][j][-1],GeneticProbability3)
        assert list(light['models'][j][0].feature_names_in_)==list(f)
        assert list(lr['models'][j][0].feature_names_in_)==list(f)
        assert list(rf['models'][j][0].feature_names_in_)==list(f)
        model=light['models'][j][-1];assert levels(model.tree_)<=3 and model.max_levels_seen_<=3
        xt=light['models'][j][0].transform(f[tr]);target=(d.loc[tr,b.g.TARGET].to_numpy()>0).astype(int)
        probs=light['models'][j].predict_proba(f[tr]);assert np.allclose(probs.sum(axis=1),1) and np.isfinite(probs).all()
        raw=np.clip(evaluate(model.tree_,xt),-30,30);assert np.allclose(probs[:,1],expit(raw))
        weight=model.class_weight_[target];weight/=weight.mean()
        fitness=np.mean(weight*(np.logaddexp(0,raw)-target*raw))+model.parsimony*size(model.tree_)
        assert np.isclose(fitness,model.fitness_)
        if j>0:assert np.allclose(light['models'][j][0][1].mean_,light['models'][j][0][0].transform(f[tr]).mean(axis=0))
    # Idénticas filas/folds y clases verdaderas para las tres familias.
    o=pd.read_csv(b.OUT/'entradas_OOF.csv',float_precision='round_trip');groups=b.g.site_ids(d[tr]).to_numpy()
    for path in [b.h.OUT,b.common.OUT]:
        old=pd.read_csv(path/'entradas_OOF.csv');assert np.array_equal(o.fila_excel,old.fila_excel) and np.array_equal(o.fold,old.fold)
        assert np.array_equal(o.presencia_real,old.presencia_real)
    for k in o.fold.unique():assert not set(groups[o.fold==k])&set(groups[o.fold!=k])
    p=o[['p_'+name for name in b.NAMES]].to_numpy();assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    assert np.array_equal(o.codigo_real_referencia,b.c.truth_codes(o.presencia_real,b.h.initial_decision(p)))
    assert bundle['bnn'].mu[0].shape==(3,8) and bundle['bnn'].mu[2].shape==(8,4)
    # Inferencia por catálogo, sin columna de incidencia, incluye las2regresiones.
    ix=rng.choice(d.index[te],96,replace=False);sample=d.loc[ix].drop(columns=[b.g.TARGET]).reset_index(drop=True)
    from sistema_integrado_roya import predict_profile
    pred=predict_profile('tres_GP_HR_red_bayesiana',sample);saved=pd.read_csv(b.OUT/'predicciones.csv',float_precision='round_trip')
    expected=saved.set_index('fila_excel').loc[d.loc[ix,'fila_excel']]
    for col in ['probabilidad_roya','posterior_inferior95','posterior_superior95','incidencia_lineal_multiple','incidencia_log1p_lineal','incidencia_predicha']:
        assert np.allclose(pred[col],expected[col],atol=1e-6,rtol=1e-6),col
    assert np.array_equal(pred.presencia_predicha,expected.presencia_predicha)
    assert np.allclose(pred[['P_FN_inicial','P_TN_inicial','P_TP_inicial','P_FP_inicial']].sum(axis=1),1)
    comparisons=pd.read_csv(b.OUT/'comparacion.csv')
    for tag in ['validacion','prueba']:
        frame=saved[saved.conjunto==tag];stats=b.common.metrics(frame[b.g.TARGET].to_numpy()>0,frame.presencia_predicha.to_numpy(bool))
        row=comparisons[(comparisons.conjunto==tag)&(comparisons.modelo=='GP_BNN')].iloc[0]
        for key in ['TN','FP','FN','TP','errores']:assert stats[key]==row[key]
    expr=pd.read_csv(b.OUT/'expresiones.csv');assert len(expr)==18 and expr.niveles.max()<=3
    result=dict(estado='aprobado',pruebas=['GP evolutiva reproducible y aprende patrón sintético','Fórmulas Bernoulli y fitness balanceado exactos',
        '18ajustes, máximo3niveles','Mismas variables, filas, clases y folds en RF/LR/GP','Misma BNN y protocolo de selección',
        'Escalado sólo train','96inferencias sin incidencia y dos regresiones','Matrices reproducidas','Referencia compartida protegida por hash'])
    (b.OUT/'verificacion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':run()
