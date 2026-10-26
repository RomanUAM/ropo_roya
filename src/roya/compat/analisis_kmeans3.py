"""Análisis exploratorio de tres grupos, sin cambiar el clasificador operativo."""
import json
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,splitter,TARGET
from ventana15_roya import features
from experimento_15dias import ROOT,predict

OUT=ROOT/'analisis_kmeans3'

def assign(b,f):
    z=b['preprocessor'].transform(f);dist=b['kmeans'].transform(z)
    ordered=np.sort(dist,axis=1);margin=(ordered[:,1]-ordered[:,0])/np.maximum(ordered[:,1],1e-12)
    group=dist.argmin(axis=1)
    return group,margin,z

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    source=joblib.load(ROOT/'evaluacion_confirmador_red6/modelo.joblib')
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);f=features(d,source['bank']);tr=split=='train';y=d[TARGET].to_numpy()>0
    prep=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),PCA(n_components=.95,svd_solver='full'))
    z=prep.fit_transform(f[tr]);km=KMeans(n_clusters=3,n_init=30,random_state=42).fit(z)
    b={'preprocessor':prep,'kmeans':km,'bank':source['bank']}
    group,margin,allz=assign(b,f)
    rates={int(j):float(y[tr][km.labels_==j].mean()) for j in range(3)}
    order=sorted(rates,key=rates.get);rank={order[0]:'menor_incidencia',order[1]:'incidencia_intermedia',order[2]:'mayor_incidencia'}
    # Anclas descriptivas, no puntos de corte biológicos ni probabilidades inferidas.
    interpretation={j:('predominio_negativo' if rates[j]<=.2 else ('predominio_positivo' if rates[j]>=.8 else 'mixto')) for j in rates}
    b.update(training_rates=rates,rank=rank,interpretation=interpretation,margin_threshold=.15,assignment=assignment)
    summary=[]
    for part in ['train','validacion','prueba']:
        for j in range(3):
            mask=(split==part)&(group==j);n=int(mask.sum());p=int(y[mask].sum())
            summary.append({'conjunto':part,'grupo':j+1,'orden':rank[j],'interpretacion_train':interpretation[j],
                            'registros':n,'positivos':p,'negativos':n-p,'fraccion_positiva':p/n if n else None,
                            'frontera_geometrica':int((mask&(margin<=.15)).sum())})
    result=d[['Municipio','Parcela','Planta','Mes_registrado','Muestreo','Coordenada_X','Coordenada_Y']].copy()
    result['conjunto']=split;result['grupo']=group+1;result['orden']=[rank[int(j)] for j in group]
    result['interpretacion_train']=[interpretation[int(j)] for j in group];result['margen_geometrico']=margin
    result['difuso_geometrico']=margin<=.15;result['presencia_observada']=y
    # Error operativo sólo descrito, nunca usado para ajustar K-means.
    predictions=predict(source,d);pred=predictions.presencia_predicha.to_numpy()
    result['presencia_modelo']=pred;result['error_modelo']=np.where(y,np.where(pred,'TP','FN'),np.where(pred,'FP','TN'))
    errors=result.groupby(['conjunto','grupo','difuso_geometrico','error_modelo']).size().rename('n').reset_index()
    result.to_csv(OUT/'puntos_grupos.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'resumen_grupos.csv',index=False)
    errors.to_csv(OUT/'errores_por_grupo_y_frontera.csv',index=False)
    stability=[]
    for seed in [7,21,84]:
        other=KMeans(3,n_init=20,random_state=seed).fit(z)
        stability.append({'seed':seed,'ARI_train':float(adjusted_rand_score(km.labels_,other.labels_))})
    report={'metodo':'Imputación, escalado y ACP95% ajustados en train; KMeans k=3 sólo train, semilla 42.',
            'componentes':z.shape[1],'silhouette_train':float(silhouette_score(z,km.labels_,sample_size=2500,random_state=42)),
            'tasas_train':rates,'orden_train':rank,'interpretacion_train':interpretation,'resumen':summary,
            'estabilidad_inicializaciones':stability,'frontera':'(d2-d1)/d2 <=0.15; distancias a dos centroides más cercanos.',
            'limites':['Tres grupos fue una hipótesis solicitada; no se demuestra que existan tres clases naturales.',
                       'No se usa incidencia para ajustar ni transformar predictores; tasas train sólo interpretan grupos después.',
                       'Frontera geométrica no equivale a probabilidad de roya ni incertidumbre clínica.',
                       'Registros por planta y quincena: un sitio puede cambiar de grupo; no son 57,600 ubicaciones independientes.',
                       'Ceros 17–24 sin confirmar, prueba consultada y mismas parcelas/fechas.',
                       'No se modifica clasificador ni se entrena árbol difuso.']}
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));joblib.dump(b,OUT/'modelo.joblib',compress=3)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(42);ix=rng.choice(np.flatnonzero(split=='prueba'),size=5000,replace=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for j in range(3):
        sample=ix[group[ix]==j];axes[0].scatter(allz[sample,0],allz[sample,1],s=6,alpha=.45,label=f'Grupo {j+1}: {rank[j]}')
    ambiguous=ix[margin[ix]<=.15];axes[0].scatter(allz[ambiguous,0],allz[ambiguous,1],s=18,facecolors='none',edgecolors='black',linewidths=.4,label='Frontera geométrica')
    for flag,color,label in [(False,'steelblue','Sin roya según base'),(True,'darkorange','Con roya según base')]:
        sample=ix[y[ix]==flag];axes[1].scatter(allz[sample,0],allz[sample,1],s=6,alpha=.45,c=color,label=label)
    for ax in axes:ax.set(xlabel='CP1',ylabel='CP2');ax.legend(fontsize=7)
    axes[0].set_title('K-means y proximidad a frontera');axes[1].set_title('Etiquetas observadas, fuera del agrupamiento')
    fig.suptitle('Muestra de 5,000 registros de prueba · proyección de dos componentes');fig.savefig(OUT/'grupos_y_etiquetas.png',dpi=180);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':run()
