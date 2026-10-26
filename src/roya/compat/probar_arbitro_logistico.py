"""Red y bosque; logística arbitra desacuerdos; regresión por grupo."""
import sys,json,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,mean_absolute_error,confusion_matrix
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features,candidate,amount,TARGET

def cycle(df):
    m=pd.to_datetime(df.Mes_registrado).dt.month.to_numpy()
    a=2*np.pi*(m-1)/12
    return np.column_stack([np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a)])

def predict(bundle,df):
    base=bundle['consenso'];f=features(df);X=base['presence_preprocess'].transform(f)
    neural=base['red'].predict(X);forest=base['forest'].predict(X)
    arbiter=bundle.get('arbitro',bundle.get('logistica'))
    p=arbiter.predict_proba(X)[:,1];disagree=neural!=forest
    presence=np.where(disagree,p>=bundle['threshold'],neural>0)
    group=np.where(neural>0,neural,forest);group=np.where(presence,group,0)
    assert (group[presence]>0).all()
    Z=base['cantidad_bundle']['positive_preprocess'].transform(f);S=cycle(df)
    result=np.zeros(len(df));state=np.full(len(df),'sin_roya',dtype=object)
    for j in range(base['cantidad_bundle']['clusters'].n_clusters):
        mask=group==j+1
        if not mask.any():continue
        name=bundle['selected'].get(j)
        if name is None:result[mask]=np.nan;state[mask]='grupo_sin_modelo_validado'
        else:
            x=np.column_stack([Z[mask],S[mask]]) if name=='ciclico' else Z[mask]
            model=bundle['models'][j][name]
            result[mask]=np.maximum(1,model.predict(x)) if name=='ciclico' else amount(model,name,x)
            state[mask]='cantidad_estimada'
    return pd.DataFrame({'grupo_red':neural,'grupo_bosque':forest,'desacuerdo':disagree,
        'probabilidad_arbitro':p,'presencia_predicha':presence,'grupo_final':group,
        'incidencia_predicha':result,'estado':state})

def main(source,previous,destination,neural=False):
    threadpool_limits(2);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    b=joblib.load(previous);base=b['cantidad_bundle'];pp=b['presence_preprocess'];cp=base['positive_preprocess'];km=base['clusters']
    d=pd.DataFrame(json.load(open(source))['registros']);f=features(d);y=d[TARGET].to_numpy()
    order=d.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    tr=order<=12;va=(order>=13)&(order<=14);te=order>=15
    estimator=(MLPClassifier(hidden_layer_sizes=(3,9,27),alpha=1,max_iter=500,
        early_stopping=False,random_state=42) if neural else LogisticRegression(C=1,max_iter=1000))
    with warnings.catch_warnings(record=True) as arbiter_warnings:
        warnings.simplefilter('always')
        log=estimator.fit(pp.transform(f[tr]),y[tr]>0)
    v=pp.transform(f[va]);p=log.predict_proba(v)[:,1]
    disagreement=b['red'].predict(v)!=b['forest'].predict(v)
    thresholds=np.arange(.05,.951,.01)
    # El árbitro se ajusta en el subconjunto de desacuerdos de validación.
    eligible=disagreement if np.unique((y[va]>0)[disagreement]).size==2 else np.ones(va.sum(),dtype=bool)
    scores=[balanced_accuracy_score((y[va]>0)[eligible],(p>=t)[eligible]) for t in thresholds]
    threshold=float(thresholds[np.argmax(scores)])
    pos=tr&(y>0);A=cp.transform(f[pos]);B=cp.transform(f[va]);g=km.predict(A);h=km.predict(B)
    S=cycle(d);models={};selected={};groups=[]
    for j in range(km.n_clusters):
        fit=g==j;valid=(h==j)&(y[va]>0);models[j]={};metrics={};notices={}
        for name in ['lineal','log1p_lineal','polinomio_2','polinomio_3','ciclico','red_10_10_10']:
            a=np.column_stack([A[fit],S[pos][fit]]) if name=='ciclico' else A[fit]
            z=np.column_stack([B[valid],S[va][valid]]) if name=='ciclico' else B[valid]
            target=np.log1p(y[pos][fit]) if name=='log1p_lineal' else y[pos][fit]
            estimator=make_pipeline(StandardScaler(),Ridge(alpha=10)) if name=='ciclico' else candidate(name)
            with warnings.catch_warnings(record=True) as caught:
                model=estimator.fit(a,target)
            models[j][name]=model;notices[name]=[str(w.message) for w in caught]
            if valid.any():
                q=np.maximum(1,model.predict(z)) if name=='ciclico' else amount(model,name,z)
                metrics[name]=float(mean_absolute_error(y[va][valid],q))
        clean={n:e for n,e in metrics.items() if not notices[n]}
        if clean:selected[j]=min(clean,key=clean.get)
        groups.append(dict(grupo=j+1,positivos_train=int(fit.sum()),positivos_validacion=int(valid.sum()),
            errores_validacion=metrics,seleccion=selected.get(j),advertencias=notices))
    bundle=dict(consenso=b,arbitro=log,threshold=threshold,models=models,selected=selected)
    report=dict(regla='Acuerdo: conservar clase. Desacuerdo de clase: arbitro decide presencia; grupo red si positivo, bosque si red cero.',
        tipo_arbitro='red_3_9_27' if neural else 'logistica',
        advertencias_arbitro=[str(w.message) for w in arbiter_warnings],
        umbral_arbitro=threshold,desacuerdos_validacion=int(disagreement.sum()),
        subconjunto_ajuste_umbral='desacuerdos' if eligible is disagreement else 'validacion_completa',
        modelo_ciclico='beta0 + beta*PC + a1*sin(mes)+b1*cos(mes)+a2*sin(2*mes)+b2*cos(2*mes), periodo anual',
        grupos=groups,limitacion='Prueba ya consultada; los errores de los clasificadores no son independientes; solo un ciclo anual.')
    for name,mask in [('validacion',va),('prueba',te)]:
        pred=predict(bundle,d[mask]);tn,fp,fn,tp=confusion_matrix(y[mask]>0,pred.presencia_predicha,labels=[0,1]).ravel()
        resolved=pred.incidencia_predicha.notna().to_numpy()
        report[name]=dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
            sensibilidad=float(tp/(tp+fn)),tasa_falsas_alarmas=float(fp/(fp+tn)),
            arbitrajes=int(pred.desacuerdo.sum()),sin_modelo_validado=int((~resolved).sum()),
            MAE_solo_resueltos_no_total=float(mean_absolute_error(y[mask][resolved],pred.loc[resolved,'incidencia_predicha'])))
        assert (pred.loc[~pred.presencia_predicha,'incidencia_predicha']==0).all()
        pred['incidencia_observada']=y[mask];pred.to_csv(out/('predicciones_'+name+'.csv'),index=False)
    joblib.dump(bundle,out/'arbitro.joblib',compress=3)
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main(*sys.argv[1:4],neural='--neuronal' in sys.argv)
