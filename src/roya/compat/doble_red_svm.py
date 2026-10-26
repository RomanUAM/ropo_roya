"""Dos redes multiclase; SVM lineal arbitra cualquier discrepancia de grupo."""
import json,sys,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,mean_absolute_error
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features,amount,TARGET

def predict(bundle,df):
    x=bundle['presence_preprocess'].transform(features(df))
    a=bundle['red_5x5'].predict(x);b=bundle['red_3_9_27'].predict(x)
    disagreement=a!=b;final=a.copy()
    arbiter=bundle.get('arbitro',bundle.get('svm'))
    if disagreement.any():final[disagreement]=arbiter.predict(x[disagreement])
    z=bundle['positive_preprocess'].transform(features(df))
    quantity=np.zeros(len(df));state=np.full(len(df),'sin_roya',dtype=object)
    for j in range(bundle['clusters'].n_clusters):
        mask=final==j+1
        if not mask.any():continue
        name=bundle['selected'].get(j)
        if name is None:quantity[mask]=np.nan;state[mask]='grupo_sin_modelo_validado'
        else:
            if name=='ciclico':
                from probar_arbitro_logistico import cycle
                v=np.column_stack([z[mask],cycle(df)[mask]])
                quantity[mask]=np.maximum(1,bundle['models'][j][name].predict(v))
            else:quantity[mask]=amount(bundle['models'][j][name],name,z[mask])
            state[mask]='cantidad_estimada'
    return pd.DataFrame({'grupo_red_5x5':a,'grupo_red_3_9_27':b,
        'desacuerdo_grupo':disagreement,
        ('intervino_arbitro' if 'arbitro' in bundle else 'intervino_svm'):disagreement,
        'grupo_final':final,'presencia_predicha':final>0,
        'incidencia_predicha':quantity,'estado':state})

def main(source,reference,destination):
    threadpool_limits(2);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    prior=joblib.load(reference);old=prior['consenso']['cantidad_bundle']
    pp=prior['consenso']['presence_preprocess'];cp=old['positive_preprocess'];km=old['clusters']
    df=pd.DataFrame(json.load(open(source))['registros']);f=features(df);y=df[TARGET].to_numpy()
    order=df.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    tr=order<=12;va=(order>=13)&(order<=14);te=order>=15
    X=pp.transform(f[tr]);labels=np.zeros(tr.sum(),dtype=int);positive=y[tr]>0
    labels[positive]=km.predict(cp.transform(f[tr][positive]))+1
    fitted={};notices={}
    estimators={
        'red_5x5':MLPClassifier(hidden_layer_sizes=(5,5,5,5,5),alpha=1,max_iter=500,early_stopping=False,random_state=42),
        'red_3_9_27':MLPClassifier(hidden_layer_sizes=(3,9,27),alpha=1,max_iter=500,early_stopping=False,random_state=42),
        'svm':LinearSVC(C=1,dual='auto',max_iter=10000,random_state=42)}
    for name,model in estimators.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X,labels)
        fitted[name]=model;notices[name]=[str(w.message) for w in caught]
        print(name,notices[name],flush=True)
    bundle=dict(**fitted,presence_preprocess=pp,positive_preprocess=cp,clusters=km,
        models=prior['models'],selected=prior['selected'])
    report={'regla':'Si las clases coinciden conservar; si son distintas usar la clase multiclase de SVM, incluido cero.',
        'redes':[[5,5,5,5,5],[3,9,27]],'svm':'LinearSVC C=1; sin probabilidad ni umbral adicional',
        'advertencias':notices,'grupos':6,
        'limitacion':'Prueba previamente consultada, grupos algoritmicos, un ciclo anual; discrepancias no entrenadas como problema separado.'}
    for name,mask in [('validacion',va),('prueba',te)]:
        p=predict(bundle,df[mask]);truth=y[mask]>0;stats={}
        for label,pred in [('red_5x5',p.grupo_red_5x5>0),('red_3_9_27',p.grupo_red_3_9_27>0),('sistema',p.presencia_predicha)]:
            tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
            stats[label]=dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
                sensibilidad=float(tp/(tp+fn)),tasa_falsas_alarmas=float(fp/(fp+tn)))
        stats['desacuerdos_grupo']=int(p.desacuerdo_grupo.sum())
        stats['desacuerdos_ambos_positivos']=int(((p.grupo_red_5x5>0)&(p.grupo_red_3_9_27>0)&p.desacuerdo_grupo).sum())
        stats['sin_modelo_validado']=int(p.incidencia_predicha.isna().sum())
        assert np.array_equal(p.intervino_svm,p.grupo_red_5x5!=p.grupo_red_3_9_27)
        agree=~p.intervino_svm
        assert np.array_equal(p.loc[agree,'grupo_final'],p.loc[agree,'grupo_red_5x5'])
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
        report[name]=stats;p['incidencia_observada']=y[mask]
        p.to_csv(out/('predicciones_'+name+'.csv'),index=False)
    joblib.dump(bundle,out/'doble_red_svm.joblib',compress=3)
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
