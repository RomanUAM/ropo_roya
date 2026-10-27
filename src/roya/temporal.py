"""Ensayo de pronóstico a un paso, aumento por bloques y evaluación hacia adelante."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.fusion_modal import PROJECT
from roya.registro import write
KEYS=['Municipio','Parcela','Coordenada_X','Coordenada_Y']

def lag_frame(d,target,dates,climate,stages,static):
    f=d.copy();f['fecha']=pd.to_datetime(dates);f=f.sort_values(KEYS+['fecha']).reset_index(drop=True)
    if f.duplicated(KEYS+['fecha']).any():raise ValueError('Serie/fecha no única')
    g=f.groupby(KEYS,sort=False);x=f[static].copy();x['roya_previa']=g[target].shift(1);x['roya_previa_2']=g[target].shift(2);x['fecha_origen']=g.fecha.shift(1);x['delta_dias']=(f.fecha-x.fecha_origen).dt.days
    for c in climate+stages:x[c+'_previo']=g[c].shift(1)
    x['mes_sen']=np.sin(2*np.pi*(f.fecha.dt.month-1)/12);x['mes_cos']=np.cos(2*np.pi*(f.fecha.dt.month-1)/12);keep=x.fecha_origen.notna();meta=f[['fila_excel','fecha',target]+KEYS].copy();meta['origen']=x.fecha_origen;meta['serie']=g.ngroup();meta=meta.loc[keep].reset_index(drop=True);x=x.loc[keep].drop(columns='fecha_origen').reset_index(drop=True)
    assert (meta.origen<meta.fecha).all();return x,meta

def augment_blocks(meta,y,seed=42):
    """Repite bloques observados de train; no fabrica valores ni ventanas nuevas."""
    rng=np.random.default_rng(seed);labels=np.ceil(y/3).astype(int);counts=np.bincount(labels);extra=[];pos={int(s):np.flatnonzero(meta.serie.to_numpy()==s) for s in meta.serie.unique()};where={int(i):(ix,j) for ix in pos.values() for j,i in enumerate(ix)}
    for c in np.unique(labels):
        if c==0:continue
        indices=np.flatnonzero(labels==c);n=min(int(counts.max()),len(indices)*3)-len(indices)
        for center in rng.choice(indices,n,replace=True):
            ix,j=where[int(center)];extra.extend(ix[max(0,j-1):min(len(ix),j+2)])
    return np.r_[np.arange(len(meta)),np.asarray(extra,int)]

def score(y,p):
    return dict(n=len(y),R2=float(r2_score(y,p)) if np.var(y)>0 else None,MAE=float(mean_absolute_error(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),prediccion_media=float(np.mean(p)))

def run():
    threadpool_limits(2);q=load();d,_,_,_=q.data();static=list(dict.fromkeys(q.e.topo_columns()+q.g.GEO+['Dist_edificio_cartografiado_m']));x,m=lag_frame(d,q.g.TARGET,q.h.reference_dates(d),q.old.c.CLIMATE,q.g.STAGES,static);y=m[q.g.TARGET].to_numpy(float);dates=np.sort(m.fecha.unique());a=round(.6*len(dates));b=round(.8*len(dates));blocks=[('principal',a,b,len(dates))]+[('retrospectivo_'+str(i),i,i+2,i+2) for i in [6,8,10]];rows=[];protocol=[]
    for name,start,end,stop in blocks:
        train=m.fecha.isin(dates[:start]).to_numpy();valid=m.fecha.isin(dates[start:end]).to_numpy();test=m.fecha.isin(dates[end:stop]).to_numpy() if name=='principal' else valid
        assert m.loc[train,'fecha'].max()<m.loc[valid,'fecha'].min()
        if name=='principal':assert m.loc[valid,'fecha'].max()<m.loc[test,'fecha'].min()
        xt=x.loc[train].reset_index(drop=True);mt=m.loc[train].reset_index(drop=True);yt=y[train];indices=augment_blocks(mt,yt);models={};candidates=[]
        for loss in ['squared_error','poisson']:
            for aug in [False,True]:
                fit=indices if aug else np.arange(len(yt));model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),HistGradientBoostingRegressor(loss=loss,max_iter=150,max_leaf_nodes=15,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=42)).fit(xt.iloc[fit],yt[fit]);label=loss+('_bloques' if aug else '_original');models[label]=model;candidates.append(dict(modelo=label,**score(y[valid],model.predict(x.loc[valid]))))
        chosen=min(candidates,key=lambda r:r['MAE'])['modelo'] if name=='principal' else None;protocol.append(dict(ensayo=name,train_fechas=start,validacion_fechas=end-start,prueba_fechas=stop-end if name=='principal' else 2,n_train=int(train.sum()),n_train_aumentado=len(indices),n_validacion=int(valid.sum()),n_prueba=int(test.sum()),positivos_prueba=int((y[test]>0).sum()),seleccion_validacion=chosen))
        for label,model in models.items():
            p=model.predict(x.loc[test]);rows.append(dict(ensayo=name,modelo=label,seleccionado=label==chosen,**score(y[test],p)))
        for label,p in [('persistencia',x.loc[test,'roya_previa'].to_numpy()),('cero',np.zeros(test.sum()))]:rows.append(dict(ensayo=name,modelo=label,seleccionado=False,**score(y[test],p)))
        print(name,chosen,flush=True)
    out=PROJECT/'reports/temporal';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'comparacion.csv',index=False);pd.DataFrame(protocol).to_csv(out/'cortes.csv',index=False);aggregate=m.assign(positiva=y>0).groupby('fecha').agg(n=('positiva','size'),positivos=('positiva','sum'));aggregate.to_csv(out/'soporte_temporal.csv');write(out/'protocolo.json',dict(ausencias_finales='Reales, confirmado por usuario. No se sustituyen ni aumentan en validación/prueba.',series=2400,identidad=KEYS,horizonte='Un muestreo adelante; roya y clima/fenología previos del mismo sitio. Mes objetivo conocido, topografía constante.',primera_observacion='Inicialización excluida: no se inventa roya previa cero. Quedan23fechas.',split='14/4/5 de23fechas elegibles (60.87/17.39/21.74), cortes acumulados redondeados al60%y80%; sin dividir una fecha.',evaluacion='Principal: validación selecciona porMAE. Prueba final toda en cero: R2indefinido. Retrospectivos son diagnósticos, no nueva prueba independiente.',retrospectivos='Entrena con6/8/10primeras fechas; evalúa dos siguientes. No se selecciona modelo en cortes retrospectivos; se informan todos los candidatos.',aumento='Bootstrap de bloques observados de hasta3registros dentro de cada serie train; centros de clases positivas se sobremuestrean hasta3veces su frecuencia, acotada por clase mayoritaria. No hay datos sintéticos nuevos ni muestras independientes adicionales.',preprocesado='Imputador ajustado en train/aumento únicamente. Sin grafos/comunidades previos que utilizaron fechas futuras. Early stopping interno desactivado.',familia='Primer control de regresión temporal HistGradientBoosting, pérdidas cuadrática/Poisson con ysin aumento; no reentrena todavía las cuatro arquitecturas multiclase.',objetivo='Incidencia porcentual, no conteo de hojas. Poisson se usa como cuasipérdida.',ley_potencias='Desbalance observado; no se ha establecido una ley de potencias.',limites='Último bloque sin positivos no valida predicción de severidad. Evaluación a un paso usa mediciones reales previas actualizadas, no pronóstico recursivo de todo el bloque. Fechas1/16aproximadas.'))
    text='# Ensayo temporal con rezagos y aumento\n\nAusencias finales confirmadas como reales. Se conserva la población y su distribución temporal.\n\n'+pd.DataFrame(protocol).to_markdown(index=False)+'\n\n'+pd.DataFrame(rows).to_markdown(index=False)+'\n\nLa prueba final no contiene positivos: R² no está definido. Predecir cero es una referencia de error cero en ese periodo; no demuestra capacidad para anticipar reaparición. Los cortes retrospectivos permiten examinar periodos positivos, pero son exploratorios. El aumento repite secuencias observadas de entrenamiento y no crea evidencia independiente. No se afirma una ley de potencias. Se informa toda la comparación, sin promover automáticamente ningún modelo.\n'
    (out/'INFORME.md').write_text(text);return rows
