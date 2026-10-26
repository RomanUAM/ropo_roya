"""Consenso positivo ANN6 AND RF6; la ANN determina el grupo de cantidad."""
import json,sys,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,mean_absolute_error
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features,amount,TARGET

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
        sensibilidad=float(tp/(tp+fn)) if tp+fn else None,
        tasa_falsas_alarmas=float(fp/(tn+fp)) if tn+fp else None)

def predict(bundle,df):
    f=features(df);x=bundle['presence_preprocess'].transform(f)
    neural=bundle['red'].predict(x);forest=bundle['forest'].predict(x)
    positive=(neural>0)&(forest>0)
    quantity=np.zeros(len(df));state=np.full(len(df),'consenso_no_positivo',dtype=object)
    cp=bundle['cantidad_bundle']['positive_preprocess'];z=cp.transform(f)
    base=bundle['cantidad_bundle']
    for j in range(base['clusters'].n_clusters):
        mask=positive&(neural==j+1)
        if not mask.any():continue
        name=base['selected'].get(j)
        if name is None:
            quantity[mask]=np.nan;state[mask]='grupo_sin_modelo_validado'
        else:
            quantity[mask]=amount(base['models'][j][name],name,z[mask])
            state[mask]='cantidad_estimada'
    return pd.DataFrame({'grupo_red':neural,'grupo_bosque':forest,
        'presencia_consenso':positive,'desacuerdo_presencia':(neural>0)!=(forest>0),
        'desacuerdo_grupo_positivo':positive&(neural!=forest),
        'grupo_final':np.where(positive,neural,0),
        'incidencia_predicha':quantity,'estado':state})

def main(source,base_path,destination):
    threadpool_limits(2);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    previous=joblib.load(base_path);base=previous['cantidad_bundle']
    d=pd.DataFrame(json.load(open(source))['registros']);f=features(d)
    y=d[TARGET].to_numpy();order=d.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    tr=order<=12;va=(order>=13)&(order<=14);te=order>=15
    pp=previous['presence_preprocess'];x=pp.transform(f[tr])
    labels=np.zeros(tr.sum(),dtype=int);positive=y[tr]>0
    labels[positive]=base['clusters'].predict(base['positive_preprocess'].transform(f[tr][positive]))+1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        red=MLPClassifier(hidden_layer_sizes=(5,5,5),alpha=1,max_iter=500,
            early_stopping=False,random_state=42).fit(x,labels)
    bundle=dict(red=red,forest=previous['forest'],presence_preprocess=pp,cantidad_bundle=base)
    report=dict(regla='presencia=(red>0 AND bosque>0); grupo=red; cero si cualquiera indica cero',
        advertencias=[str(w.message) for w in caught],
        limitacion='Comparacion exploratoria: prueba ya consultada; modelos comparten datos, sus errores no son independientes.')
    for name,mask in [('validacion',va),('prueba',te)]:
        p=predict(bundle,d[mask]);truth=y[mask]>0
        a=metrics(truth,p.grupo_red.to_numpy()>0);b=metrics(truth,p.grupo_bosque.to_numpy()>0)
        consensus=metrics(truth,p.presencia_consenso.to_numpy())
        assert consensus['FP']<=min(a['FP'],b['FP'])
        assert consensus['TP']<=min(a['TP'],b['TP'])
        assert (p.loc[~p.presencia_consenso,'incidencia_predicha']==0).all()
        resolved=p.incidencia_predicha.notna().to_numpy()
        report[name]=dict(red_multiclase=a,bosque_multiclase=b,consenso=consensus,
            desacuerdos_presencia=int(p.desacuerdo_presencia.sum()),
            desacuerdos_grupo_positivo=int(p.desacuerdo_grupo_positivo.sum()),
            cantidad_sin_modelo_validado=int((~resolved).sum()),
            MAE_solo_resueltos_no_total=float(mean_absolute_error(y[mask][resolved],p.loc[resolved,'incidencia_predicha'])))
        p['incidencia_observada']=y[mask]
        for col in ['Municipio','Parcela','Mes_registrado','Muestreo']:p[col]=d.loc[mask,col].to_numpy()
        p.to_csv(out/('predicciones_'+name+'.csv'),index=False)
    old=base['gate'].predict_proba(pp.transform(f[te]))[:,1]>=base['threshold']
    report['referencia_red_binaria_anterior']=metrics(y[te]>0,old)
    joblib.dump(bundle,out/'consenso.joblib',compress=3)
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
