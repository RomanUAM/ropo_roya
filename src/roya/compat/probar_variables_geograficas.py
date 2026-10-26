"""Variante exploratoria con distancias recuperadas del mapa previo."""
import json,sys,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,accuracy_score
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features,preprocess,amount,TARGET

GEO=['Dist_cauce_cartografiado_m','Dist_camino_cartografiado_m','Dist_zona_residencial_cartografiada_m']

def add_geo(bundle,df):
    if all(c in df for c in GEO):return df.copy()
    # Redondeo explícito a 1 micrómetro para absorber serialización CSV/JSON;
    # nunca se busca un vecino espacial para sustituir un sitio desconocido.
    if any(c in df for c in GEO):raise ValueError('Columnas geograficas incompletas: se requieren las tres distancias.')
    left=df.copy();right=bundle['distancias'].copy()
    for key,col in [('_gx','Coordenada_X'),('_gy','Coordenada_Y')]:
        left[key]=left[col].round(6);right[key]=right[col].round(6)
    merged=left.merge(right[['_gx','_gy']+GEO].drop_duplicates(['_gx','_gy']),on=['_gx','_gy'],how='left',validate='many_to_one',sort=False).drop(columns=['_gx','_gy'])
    if merged[GEO].isna().any().any():
        raise ValueError('Hay sitios sin distancia calculada: enriquecer primero con las capas y CRS confirmado.')
    return merged

def predict(bundle,df):
    df=add_geo(bundle,df);base=bundle['base'];f=features(df);x=pd.concat([f,df[GEO]],axis=1)
    p=bundle['gate'].predict_proba(bundle['presence_preprocess'].transform(x))[:,1]
    active=p>=bundle['threshold'];groups=np.zeros(len(df),dtype=int)
    if active.any():groups[active]=bundle['forest'].predict(x[active].to_numpy())
    z=base['positive_preprocess'].transform(f);quantity=np.zeros(len(df));state=np.full(len(df),'sin_roya',dtype=object)
    for j in range(base['clusters'].n_clusters):
        mask=groups==j+1
        if not mask.any():continue
        name=bundle['cantidad']['selected'].get(j)
        if name is None:quantity[mask]=np.nan;state[mask]='grupo_sin_modelo_validado'
        else:quantity[mask]=amount(bundle['cantidad']['models'][j][name],name,z[mask]);state[mask]='cantidad_estimada'
    return pd.DataFrame({'probabilidad_presencia':p,'presencia_predicha':active,'grupo_final':groups,
        'incidencia_predicha':quantity,'estado':state})

def main(csv,reference,destination):
    threadpool_limits(2);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(csv);old=joblib.load(reference);base=old['base'];qbase=old['cantidad']
    f=features(d);x=pd.concat([f,d[GEO]],axis=1);y=d[TARGET].to_numpy();o=d.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    tr=o<=12;va=(o>=13)&(o<=14);te=o>=15
    pp=preprocess();X=pp.fit_transform(x[tr]);V=pp.transform(x[va])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gate=MLPClassifier(hidden_layer_sizes=(5,5,5),alpha=1,max_iter=500,early_stopping=False,random_state=42).fit(X,y[tr]>0)
    pv=gate.predict_proba(V)[:,1];thresholds=np.arange(.05,.951,.01)
    threshold=float(thresholds[np.argmax([balanced_accuracy_score(y[va]>0,pv>=t) for t in thresholds])])
    positive=tr&(y>0);labels=base['clusters'].predict(base['positive_preprocess'].transform(f[positive]))+1
    forest=RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=20,n_jobs=2,random_state=42).fit(x[positive].to_numpy(),labels)
    lookup=d[['Coordenada_X','Coordenada_Y']+GEO].drop_duplicates(['Coordenada_X','Coordenada_Y'])
    bundle={'base':base,'cantidad':qbase,'gate':gate,'presence_preprocess':pp,'threshold':threshold,
            'forest':forest,'distancias':lookup,'geo_features':GEO,'crs':'EPSG:32614 provisional'}
    report={'variables_agregadas':GEO,'componentes_presencia':int(pp[-1].n_components_),
        'threshold':threshold,'advertencias':[str(w.message) for w in caught],
        'crs':'EPSG:32614 provisional heredado del mapa; no confirmado',
        'excluidos':['edificios: solo uno disponible','acantilado no equivale a cerro',
                     'cerros y centros urbanos verificados sin capa'],
        'limitacion':'Experimento con geometria incompleta y prueba reutilizada; no confirma causalidad ni despliegue.'}
    for name,mask in [('validacion',va),('prueba',te)]:
        p=predict(bundle,d[mask]);tn,fp,fn,tp=confusion_matrix(y[mask]>0,p.presencia_predicha,labels=[0,1]).ravel()
        pos=mask&(y>0);target=base['clusters'].predict(base['positive_preprocess'].transform(f[pos]))+1
        group_pred=forest.predict(x[pos].to_numpy())
        report[name]={'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
            'sensibilidad':float(tp/(tp+fn)),'tasa_falsas_alarmas':float(fp/(fp+tn)),
            'acierto_grupo_positivos':float(accuracy_score(target,group_pred)),
            'sin_modelo_cantidad_validado':int(p.incidencia_predicha.isna().sum())}
        p['incidencia_observada']=y[mask];p.to_csv(out/('predicciones_'+name+'.csv'),index=False)
    joblib.dump(bundle,out/'geografia.joblib',compress=3)
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
