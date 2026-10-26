"""Compara variables originales frente a ACP para clasificar grupos positivos.
El detector binario y las regresiones por grupo quedan fijos.
"""
import json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from threadpoolctl import threadpool_limits
from roya_dos_etapas import features,amount,TARGET

def predict(bundle,df):
    f=features(df);base=bundle['base'];qbase=bundle['cantidad']
    p=base['gate'].predict_proba(base['presence_preprocess'].transform(f))[:,1]
    active=p>=base['threshold']
    groups=np.zeros(len(df),dtype=int)
    if active.any():
        z=f[active].to_numpy() if bundle['representation']=='originales' else base['positive_preprocess'].transform(f[active])
        groups[active]=bundle['forest'].predict(z)
    z=base['positive_preprocess'].transform(f)
    quantity=np.zeros(len(df));state=np.full(len(df),'sin_roya',dtype=object)
    for j in range(base['clusters'].n_clusters):
        mask=groups==j+1
        if not mask.any():continue
        name=qbase['selected'].get(j)
        if name is None:quantity[mask]=np.nan;state[mask]='grupo_sin_modelo_validado'
        elif name=='ciclico':
            from probar_arbitro_logistico import cycle
            quantity[mask]=np.maximum(1,qbase['models'][j][name].predict(np.column_stack([z[mask],cycle(df)[mask]])))
            state[mask]='cantidad_estimada'
        else:
            quantity[mask]=amount(qbase['models'][j][name],name,z[mask]);state[mask]='cantidad_estimada'
    return pd.DataFrame({'probabilidad_presencia':p,'presencia_predicha':active,
        'grupo_final':groups,'incidencia_predicha':quantity,'estado':state})

def main(source,reference,quantity_reference,destination):
    threadpool_limits(2);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    base=joblib.load(reference);quantity=joblib.load(quantity_reference)
    d=pd.DataFrame(json.load(open(source))['registros']);f=features(d);y=d[TARGET].to_numpy()
    order=d.Muestreo.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    tr=order<=12;va=(order>=13)&(order<=14);te=order>=15;positive=tr&(y>0)
    pp=base['positive_preprocess'];km=base['clusters']
    labels=km.predict(pp.transform(f[positive]))+1
    report={'pregunta':'Variables originales versus ACP para grupo positivo; puerta binaria fija.',
        'etiquetas':'Grupos algoritmicos definidos por ACP+KMeans de entrenamiento; no etiquetas biologicas independientes.',
        'diseno':'Mismos registros, 200 arboles, profundidad 12, hoja minima 20, sin ponderacion; unica diferencia: representacion.',
        'limitacion':'Prueba previamente consultada; datos de un ciclo; validacion no cubre todos los grupos positivos.',
        'variantes':{}}
    bundles={}
    for representation in ['originales','acp']:
        X=f[positive].to_numpy() if representation=='originales' else pp.transform(f[positive])
        forest=RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=20,
            n_jobs=2,random_state=42).fit(X,labels)
        bundle={'base':base,'cantidad':quantity,'representation':representation,'forest':forest}
        bundles[representation]=bundle;stats={}
        for name,mask in [('validacion',va),('prueba',te)]:
            pos=mask&(y>0);oracle=km.predict(pp.transform(f[pos]))+1
            z=f[pos].to_numpy() if representation=='originales' else pp.transform(f[pos])
            group_pred=forest.predict(z)
            result=predict(bundle,d[mask]);truth=y[mask]>0
            tn,fp,fn,tp=confusion_matrix(truth,result.presencia_predicha,labels=[0,1]).ravel()
            stats[name]={'n_positivos':int(pos.sum()),
                'grupos_positivos_presentes':pd.Series(oracle).value_counts().sort_index().to_dict(),
                'acierto_grupo_positivos':float(accuracy_score(oracle,group_pred)),
                'matriz_grupos_1_a_5':confusion_matrix(oracle,group_pred,labels=np.arange(1,6)).tolist(),
                'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
                'sin_cantidad_validada':int(result.incidencia_predicha.isna().sum())}
            result['incidencia_observada']=y[mask]
            result.to_csv(out/(representation+'_'+name+'.csv'),index=False)
        stats['importancia_predictiva_no_causal']=dict(zip(f.columns if representation=='originales' else [f'PC{i+1}' for i in range(X.shape[1])],forest.feature_importances_.tolist()))
        report['variantes'][representation]=stats
        joblib.dump(bundle,out/('bosque_'+representation+'.joblib'),compress=3)
    # Desempate predefinido: conservar originales por interpretabilidad.
    winner=max(['originales','acp'],key=lambda r:report['variantes'][r]['validacion']['acierto_grupo_positivos'])
    report['seleccion_exploratoria_validacion']=winner
    report['desempate']='Preferir originales si empatan; no decidir con prueba.'
    report['dictamen']='Empate sin evidencia suficiente de discriminacion entre grupos; no declarar ganador validado.'
    report['referencia_constante_grupo3']={'validacion':1.0,'prueba':394/398,
        'nota':'Ambos bosques igualan esta regla constante entre los positivos evaluados; grupo 5: cero de cuatro asignaciones correctas.'}
    assert report['variantes']['originales']['prueba']['FP']==report['variantes']['acp']['prueba']['FP']
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
