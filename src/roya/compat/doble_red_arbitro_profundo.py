"""Dos redes iniciales y árbitro multiclase 20–15–12–9–9–12."""
import json,sys,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from doble_red_svm import predict
from roya_dos_etapas import features,TARGET

def main(source,previous,destination):
    threadpool_limits(2);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    old=joblib.load(previous);bundle={k:v for k,v in old.items() if k!='svm'}
    d=pd.DataFrame(json.load(open(source))['registros']);f=features(d);y=d[TARGET].to_numpy()
    order=d.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    tr=order<=12;va=(order>=13)&(order<=14);te=order>=15
    x=bundle['presence_preprocess'].transform(f[tr]);labels=np.zeros(tr.sum(),dtype=int)
    positive=y[tr]>0
    labels[positive]=bundle['clusters'].predict(bundle['positive_preprocess'].transform(f[tr][positive]))+1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        arbiter=MLPClassifier(hidden_layer_sizes=(20,15,12,9,9,12),alpha=1,max_iter=500,
            early_stopping=False,random_state=42).fit(x,labels)
    bundle['arbitro']=arbiter
    report={'arquitectura_arbitro':[20,15,12,9,9,12],
        'regla':'Redes iniciales 5–5–5–5–5 y 3–9–27; si clases distintas, arbitro decide clase final 0–5.',
        'advertencias':[str(w.message) for w in caught],
        'limitacion':'Arbitro entrenado con todo entrenamiento; no entrenado especificamente en discrepancias fuera de muestra. Prueba ya consultada.'}
    for name,mask in [('validacion',va),('prueba',te)]:
        p=predict(bundle,d[mask]);truth=y[mask]>0
        tn,fp,fn,tp=confusion_matrix(truth,p.presencia_predicha,labels=[0,1]).ravel()
        disagreement=p.grupo_red_5x5!=p.grupo_red_3_9_27
        assert np.array_equal(disagreement,p.intervino_arbitro)
        agree=~disagreement
        assert np.array_equal(p.loc[agree,'grupo_final'],p.loc[agree,'grupo_red_5x5'])
        assert (p.loc[~p.presencia_predicha,'incidencia_predicha']==0).all()
        report[name]=dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
            sensibilidad=float(tp/(tp+fn)),tasa_falsas_alarmas=float(fp/(tn+fp)),
            discrepancias=int(disagreement.sum()),sin_modelo_validado=int(p.incidencia_predicha.isna().sum()))
        p['incidencia_observada']=y[mask];p.to_csv(out/('predicciones_'+name+'.csv'),index=False)
    joblib.dump(bundle,out/'arbitro_profundo.joblib',compress=3)
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
