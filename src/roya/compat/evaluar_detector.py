import json, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features

def evaluate(truth, probability, threshold):
    tn,fp,fn,tp=confusion_matrix(truth, probability>=threshold,labels=[0,1]).ravel()
    return dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
        sensibilidad=float(tp/(tp+fn)) if tp+fn else None,
        tasa_falsas_alarmas=float(fp/(tn+fp)) if tn+fp else None,
        precision=float(tp/(tp+fp)) if tp+fp else None)

def threshold_for_recall(y,p,recall):
    # Umbral máximo cuya sensibilidad empírica en validación cumple el objetivo.
    values=np.sort(p[y])
    if not len(values): return None
    allowed=int(np.floor((1-recall)*len(values)+1e-9))
    return float(values[allowed])

threadpool_limits(2)
source = sys.argv[1] if len(sys.argv)>1 else 'datos/Registros_locales_roya.json'
out = Path(sys.argv[2] if len(sys.argv)>2 else '.')
out.mkdir(parents=True, exist_ok=True)
d=pd.DataFrame(json.load(open(source))['registros'])
f=features(d); y=(d['Incidencia de roya del Café']>0).to_numpy()
order=d.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
results=[]
for end in [6,8,12,14]:
    train=order<=end; val=(order>end)&(order<=end+2)
    test=order>end+2 if end==12 else ((order>end+2)&(order<=end+4))
    pp=make_pipeline(StandardScaler(),PCA(n_components=.95,svd_solver='full'))
    X=pp.fit_transform(f[train]); V=pp.transform(f[val]); T=pp.transform(f[test])
    with warnings.catch_warnings(record=True) as w:
        estimator=(LogisticRegression(C=1,max_iter=1000) if '--logistica' in sys.argv else
            MLPClassifier(hidden_layer_sizes=(5,5,5),alpha=1,max_iter=500,
            early_stopping=False,random_state=42))
        gate=estimator.fit(X,y[train])
    pv=gate.predict_proba(V)[:,1];pt=gate.predict_proba(T)[:,1]
    for goal in [.90,.95]:
        threshold=threshold_for_recall(y[val],pv,goal)
        row={'ultimo_muestreo_train':end,'test_periodos':sorted(np.unique(order[test]).tolist()),
            'objetivo_sensibilidad_validacion':goal,'umbral':threshold,
            'validacion':evaluate(y[val],pv,threshold),
            'prueba':evaluate(y[test],pt,threshold),
            'advertencias':[str(v.message) for v in w]}
        results.append(row);print(json.dumps(row),flush=True)
monthly=d.assign(presencia=y).groupby('Mes_registrado').agg(
    registros=('presencia','size'),positivos=('presencia','sum'))
monthly.to_csv(out/'auditoria_ceros_mensual.csv')
(out/'comparacion_umbrales.json').write_text(json.dumps(results,indent=2))
