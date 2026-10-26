"""Experimento alternativo: clase cero sin roya y K grupos positivos.
No sustituye el detector aprobado ni crea grupos positivos usando ceros.
"""
import json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,accuracy_score,mean_absolute_error
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features,amount,TARGET

def metrics(y,positive):
    tn,fp,fn,tp=confusion_matrix(y,positive,labels=[0,1]).ravel()
    return {'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
            'sensibilidad':float(tp/(tp+fn)) if tp+fn else None,
            'falsas_alarmas':float(fp/(tn+fp)) if tn+fp else None,
            'exactitud_balanceada':float(balanced_accuracy_score(y,positive))}

threadpool_limits(2)
source=sys.argv[1];base=sys.argv[2];out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
d=pd.DataFrame(json.load(open(source))['registros']);b=joblib.load(base)
f=features(d);y=d[TARGET].to_numpy();o=d.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
tr=o<=12;va=(o>=13)&(o<=14);te=o>=15
pp=b['presence_preprocess'];cp=b['positive_preprocess'];km=b['clusters']
X=pp.transform(f[tr]);V=pp.transform(f[va]);T=pp.transform(f[te])
# Etiquetas positivas inducidas por el agrupamiento previo de entrenamiento.
labels=np.zeros(len(d),dtype=int)
for mask in [tr,va,te]:
    positive=mask&(y>0)
    labels[positive]=km.predict(cp.transform(f[positive]))+1
rows=[];models=[]
for weight in [None,'balanced_subsample']:
    forest=RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=20,
        class_weight=weight,n_jobs=2,random_state=42).fit(X,labels[tr])
    pred=forest.predict(V)
    row={'ponderacion':weight,'validacion':metrics(y[va]>0,pred>0),
         'acierto_grupo_positivos_validacion':float(accuracy_score(labels[va][y[va]>0],pred[y[va]>0]))}
    rows.append(row);models.append(forest);print(json.dumps(row),flush=True)
chosen=int(np.argmax([r['validacion']['exactitud_balanceada'] for r in rows]));forest=models[chosen]
pred=forest.predict(T);expected=np.zeros(te.sum());states=np.full(te.sum(),'sin_roya',dtype=object)
z=cp.transform(f[te])
for group in range(km.n_clusters):
    mask=pred==group+1
    if not mask.any():continue
    name=b['selected'].get(group)
    if name is None:expected[mask]=np.nan;states[mask]='grupo_sin_validacion_positiva'
    else:expected[mask]=amount(b['models'][group][name],name,z[mask]);states[mask]='cantidad_estimada'
prob=forest.predict_proba(T);p_presence=1-prob[:,np.flatnonzero(forest.classes_==0)[0]]
result=pd.DataFrame({'clase_predicha':pred,'probabilidad_presencia':p_presence,
    'incidencia_predicha':expected,'incidencia_observada':y[te],'estado':states})
result.to_csv(out/'predicciones.csv',index=False)
neural=b['gate'].predict_proba(T)[:,1]>=b['threshold']
report={'tipo':'alternativa_experimental_no_sustituye_modelo_anterior',
    'decision':'argmax entre clase cero y grupos positivos; sin ajuste adicional de umbral',
    'etiquetas':'grupos algoritmicos previos; no son estadios biologicos verificados',
    'seleccion_exclusiva_validacion':rows,'configuracion_elegida':rows[chosen]['ponderacion'],
    'prueba_random_forest':metrics(y[te]>0,pred>0),'prueba_red_anterior':metrics(y[te]>0,neural),
    'acierto_grupo_positivos_prueba':float(accuracy_score(labels[te][y[te]>0],pred[y[te]>0])),
    'sin_modelo_cantidad_validado':int(np.isnan(expected).sum()),
    'advertencia':'Prueba previamente consultada. Una mejora aqui no confirma generalizacion a otro ciclo o parcela.'}
(out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
joblib.dump({'forest':forest,'presence_preprocess':pp,'positive_preprocess':cp,
             'positive_cluster':km,'cantidad_bundle':b},out/'alternativa_forest.joblib',compress=3)
print(json.dumps(report),flush=True)
