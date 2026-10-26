"""Originales, ACP conjunto y ACP por bloques sobre presencia histórica.
No entrena ni valida la favorabilidad de los próximos siete días.
"""
import json,sys,warnings
from pathlib import Path
import joblib,numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,inputs,splitter,binary_metrics

def run(geo,terrain,destination):
    threadpool_limits(2);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    d=assemble(geo,terrain);f=inputs(d);s,_=splitter(d);y=d['Incidencia de roya del Café'].to_numpy()>0
    tr=s=='train';va=s=='validacion';te=s=='prueba'
    climate=['Temperatura promedio','Punto_de_rocio','Temperatura','Humedad_especifica','Humedad_relativa','Precipitacion']
    vegetal=['Brotacion','Floracion','Amarre','Lechoso','Consistente','Maduro']
    month=['mes_sen','mes_cos'];territory=[c for c in f if c not in climate+vegetal+month]
    def pca():return make_pipeline(StandardScaler(),PCA(n_components=.95,svd_solver='full'))
    methods={'originales':StandardScaler(),'ACP_conjunto':pca(),
        'ACP_bloques':ColumnTransformer([('clima',pca(),climate),('vegetal',pca(),vegetal),
            ('territorio',pca(),territory),('mes',StandardScaler(),month)])}
    rows=[];saved={}
    for name,prep in methods.items():
        X=prep.fit_transform(f[tr]);V=prep.transform(f[va]);T=prep.transform(f[te])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model=MLPClassifier(hidden_layer_sizes=(6,9,12),alpha=1,max_iter=500,early_stopping=False,random_state=42).fit(X,y[tr])
        pv=model.predict_proba(V)[:,1];thresholds=np.arange(.05,.951,.01)
        threshold=float(thresholds[np.argmax([balanced_accuracy_score(y[va],pv>=t) for t in thresholds])])
        row={'representacion':name,'dimensiones':X.shape[1],'umbral':threshold,
            'validacion':binary_metrics(y[va],pv,threshold),
            'prueba':binary_metrics(y[te],model.predict_proba(T)[:,1],threshold),
            'advertencias':[str(w.message) for w in caught]}
        rows.append(row);saved[name]={'preprocessor':prep,'model':model,'threshold':threshold}
        print(json.dumps(row,ensure_ascii=False),flush=True)
    eligible=[r for r in rows if not r['advertencias']]
    best=max(eligible,key=lambda r:r['validacion']['exactitud_balanceada'])
    report={'objetivo':'Presencia histórica en plantas reservadas; no favorabilidad meteorológica diaria.',
        'particion':'50/20/30 por plantas dentro de parcelas; mismas fechas',
        'comparacion':rows,'seleccion_solo_validacion':best['representacion'],
        'clima_simulado':'No disponible y no utilizado; no se sustituyen ni duplican columnas climáticas sin correspondencia validada.',
        'limites':['Prueba previamente consultada','CRS y altitud provisionales','No valida ventanas de siete días ni parcelas nuevas']}
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    joblib.dump(saved,out/'representaciones.joblib',compress=3)

if __name__=='__main__':run(*sys.argv[1:])
