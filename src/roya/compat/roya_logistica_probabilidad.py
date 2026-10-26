"""Presencia logística y cantidad por grupo: ACP frente a ACP + probabilidad.
Las probabilidades de entrenamiento de cantidad se generan hacia adelante.
"""
import json, sys, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, confusion_matrix, brier_score_loss, balanced_accuracy_score
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features, preprocess, candidate, amount, TARGET

def logistic():
    return LogisticRegression(C=1,max_iter=1000)

def temporal_probabilities(f, y, order, training):
    periods=np.sort(np.unique(order[training]))
    probabilities=np.full(len(y),np.nan)
    # Cuatro muestreos iniciales permiten arrancar; no reciben probabilidades
    # de una red/modelo que haya visto su respuesta ni de periodos futuros.
    folds=[]
    for start in range(4,len(periods),2):
        fit=training & np.isin(order,periods[:start])
        hold=training & np.isin(order,periods[start:start+2])
        if np.unique(y[fit]>0).size<2: continue
        pp=preprocess();X=pp.fit_transform(f[fit]);Z=pp.transform(f[hold])
        model=logistic().fit(X,y[fit]>0)
        probabilities[hold]=model.predict_proba(Z)[:,1]
        assert order[fit].max()<order[hold].min()
        folds.append({'ultimo_train':int(order[fit].max()),
            'muestreos_predichos':sorted(np.unique(order[hold]).tolist()),
            'filas':int(hold.sum())})
    return probabilities,folds

def predict(bundle, df, variant='con_probabilidad'):
    f=features(df)
    p=bundle['gate'].predict_proba(bundle['presence_preprocess'].transform(f))[:,1]
    active=p>=bundle['threshold']
    z=bundle['positive_preprocess'].transform(f)
    group=bundle['clusters'].predict(z)
    a=np.column_stack([z,p]) if variant=='con_probabilidad' else z
    result=np.zeros(len(df));status=np.full(len(df),'sin_roya',dtype=object)
    for j in range(bundle['clusters'].n_clusters):
        mask=active&(group==j)
        if not mask.any():continue
        name=bundle['selected'][variant].get(j)
        if name is None:
            result[mask]=np.nan;status[mask]='grupo_sin_validacion_positiva'
        else:
            result[mask]=amount(bundle['models'][variant][j][name],name,a[mask])
            status[mask]='cantidad_estimada'
    return pd.DataFrame({'probabilidad_presencia':p,'presencia_predicha':active,
        'grupo_candidato':group+1,'incidencia_predicha':result,'estado':status})

def main(source,destination):
    threadpool_limits(2)
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(json.loads(Path(source).read_text())['registros'])
    f=features(df);y=df[TARGET].to_numpy(dtype=float)
    order=df.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    train=order<=12;val=(order>=13)&(order<=14);test=order>=15
    assert not ((train&val)|(train&test)|(val&test)).any()
    pp=preprocess();X=pp.fit_transform(f[train]);V=pp.transform(f[val])
    gate=logistic().fit(X,y[train]>0);pv=gate.predict_proba(V)[:,1]
    thresholds=np.arange(.05,.951,.01)
    scores=[balanced_accuracy_score(y[val]>0,pv>=t) for t in thresholds]
    threshold=float(thresholds[np.argmax(scores)])
    poof,folds=temporal_probabilities(f,y,order,train)
    positive=train&(y>0)&np.isfinite(poof)
    cp=preprocess();A=cp.fit_transform(f[positive]);B=cp.transform(f[val])
    best=None;search=[]
    for k in range(2,7):
        km=KMeans(k,n_init=20,random_state=42).fit(A)
        score=float(silhouette_score(A,km.labels_,sample_size=min(2500,len(A)),random_state=42))
        search.append({'k':k,'silueta':score})
        if best is None or score>best[0]:best=(score,km)
    km=best[1];g=km.labels_;h=km.predict(B)
    names=['lineal','log1p_lineal','polinomio_2','polinomio_3','red_10_10_10']
    models={};selected={};reports={}
    for variant in ['sin_probabilidad','con_probabilidad']:
        models[variant]={};selected[variant]={};groups=[]
        a=np.column_stack([A,poof[positive]]) if variant=='con_probabilidad' else A
        b=np.column_stack([B,pv]) if variant=='con_probabilidad' else B
        for j in range(km.n_clusters):
            fit=g==j;valid=(h==j)&(y[val]>0)
            models[variant][j]={};metrics={};notices={}
            for name in names:
                target=np.log1p(y[positive][fit]) if name=='log1p_lineal' else y[positive][fit]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model=candidate(name).fit(a[fit],target)
                models[variant][j][name]=model
                notices[name]=[str(w.message) for w in caught]
                if valid.any():metrics[name]=float(mean_absolute_error(y[val][valid],amount(model,name,b[valid])))
            eligible={name:error for name,error in metrics.items() if not notices[name]}
            if eligible:selected[variant][j]=min(eligible,key=eligible.get)
            row={'grupo':j+1,'positivos_train':int(fit.sum()),
                'positivos_validacion':int(valid.sum()),'MAE_validacion_positivos':metrics,
                'seleccion':selected[variant].get(j),'advertencias':notices}
            groups.append(row);print(variant,json.dumps(row),flush=True)
        reports[variant]=groups
    bundle={'gate':gate,'presence_preprocess':pp,'positive_preprocess':cp,
        'clusters':km,'threshold':threshold,'models':models,'selected':selected,
        'tipo':'logistica_con_probabilidad_temporal','folds_probabilidad':folds}
    tests={}
    for variant in reports:
        result=predict(bundle,df[test],variant);truth=y[test]
        result['incidencia_observada']=truth
        for c in ['Municipio','Parcela','Mes_registrado','Muestreo']:
            result[c]=df.loc[test,c].to_numpy()
        complete=result.incidencia_predicha.notna().to_numpy()
        tests[variant]={'matriz_confusion_TN_FP_FN_TP':confusion_matrix(truth>0,result.presencia_predicha,labels=[0,1]).tolist(),
            'Brier':float(brier_score_loss(truth>0,result.probabilidad_presencia)),
            'sin_modelo_validado':int((~complete).sum()),
            'MAE_solo_resueltos_no_comparable_con_total':float(mean_absolute_error(truth[complete],result.loc[complete,'incidencia_predicha'])),
            'MAE_referencia_cero_total':float(np.mean(truth))}
        result.to_csv(out/('predicciones_'+variant+'.csv'),index=False)
        if variant=='con_probabilidad':result.to_csv(out/'predicciones_prueba.csv',index=False)
    report={'advertencia':'Prueba previamente consultada; comparacion exploratoria, no evaluacion independiente final.',
        'detector':'regresion logistica multiple sobre ACP',
        'split':{name:{'n':int(mask.sum()),'positivos':int((y[mask]>0).sum())}
                 for name,mask in [('train',train),('validacion',val),('prueba',test)]},
        'threshold':threshold,'componentes_presencia':int(pp[-1].n_components_),
        'componentes_positivos':int(cp[-1].n_components_),
        'folds_probabilidad':folds,'positivos_cantidad_train':int(positive.sum()),
        'excluidos_arranque_cantidad':int((train&(y>0)&~np.isfinite(poof)).sum()),
        'busqueda_grupos':search,'comparacion_cantidad':reports,
        'grupos':reports['con_probabilidad'],'prueba':tests['con_probabilidad'],
        'comparacion_prueba':tests}
    joblib.dump(bundle,out/'modelos.joblib')
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'threshold':threshold,'grupos':km.n_clusters,'prueba':tests}),flush=True)

if __name__=='__main__':main(sys.argv[1],sys.argv[2])
