"""Probabilidad global calibrada OOF, dos umbrales y q condicional del confirmador."""
import json
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits
from experimento_15dias import ROOT
from ventana15_roya import features
from filtro_logistico_condicional import meta_features
from bosque_bayes_regresion import enrich
from modelo_roya_50_20_30 import assemble,splitter,TARGET

OUT=ROOT/'umbrales_condicionales'

def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)

def wilson(k,n):
    if n==0:return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;delta=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-delta),min(1,center+delta)

def find_thresholds(p,y,min_n=200):
    order=np.argsort(p);a=p[order];yy=y[order].astype(int);cs=np.cumsum(yy);n=len(a)
    ends=np.flatnonzero(np.r_[a[1:]!=a[:-1],True]);negative=[];positive=[]
    for end in ends:
        count=end+1
        if count>=min_n:
            _,upper=wilson(int(cs[end]),count)
            if upper<=.05:negative.append(float(a[end]))
    low=max(negative) if negative else None
    starts=np.r_[0,ends[:-1]+1]
    for start in starts:
        count=n-start
        if count<min_n:continue
        k=int(cs[-1]-(cs[start-1] if start else 0));lower,_=wilson(k,count)
        if lower>=.90 and (low is None or a[start]>low):positive.append(float(a[start]))
    high=min(positive) if positive else None
    return low,high

def classify(p,low,high):
    label=np.full(len(p),'indeterminado',dtype=object)
    if low is not None:label[p<=low]='negativo'
    if high is not None:label[p>=high]='positivo'
    return label

def predict(b,d):
    d=enrich(b['source'],d);f=features(d,b['source']['bank']);source=b['source']
    raw=source['primary_models'][source['selected_primary']].predict_proba(source['preprocessor'].transform(f))[:,1]
    probability=b['calibrator'].predict_proba(logit(raw))[:,1]
    selected=raw>=source['primary_thresholds'][source['selected_primary']];q=np.full(len(d),np.nan)
    if selected.any():q[selected]=source['confirmers']['red6'].predict_proba(meta_features(f.iloc[np.flatnonzero(selected)],raw[selected]))[:,1]
    return pd.DataFrame({'probabilidad_RF':raw,'probabilidad_global_calibrada':probability,
        'categoria':classify(probability,b['low'],b['high']),
        'positivo_RF':selected,'P_positivo_real_dado_RF_positivo':q,'P_falso_positivo_dado_RF_positivo':1-q})

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    source=joblib.load(ROOT/'evaluacion_confirmador_red6/modelo.joblib')
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv');split,assignment=splitter(d)
    pd.testing.assert_frame_equal(assignment,source['particion_plantas']);y=d[TARGET].to_numpy()>0
    oof=pd.read_csv(ROOT/'evaluacion_15dias/ejemplos_confirmadores_OOF.csv')
    assert len(oof)==(split=='train').sum()
    calibrator=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(logit(oof.prob_detector_OOF.to_numpy()),y[split=='train'])
    if calibrator.coef_[0,0]<=0:raise RuntimeError('Calibración no monótona: revisar detector.')
    b={'source':source,'calibrator':calibrator,'low':None,'high':None,'schema':'umbrales_condicionales_v1'}
    val=predict(b,d[split=='validacion']);raw_low,raw_high=find_thresholds(val.probabilidad_global_calibrada.to_numpy(),y[split=='validacion'])
    vp=val.probabilidad_global_calibrada.to_numpy();vy=y[split=='validacion']
    neg=vp<=.05;pos=vp>=.90
    low=.05 if neg.sum()>=200 and wilson(int(vy[neg].sum()),int(neg.sum()))[1]<=.05 else None
    high=.90 if pos.sum()>=200 and wilson(int(vy[pos].sum()),int(pos.sum()))[0]>=.90 else None
    b.update(low=low,high=high);summary=[];calibration=[]
    for part in ['train','validacion','prueba']:
        mask=split==part;result=predict(b,d[mask]);truth=y[mask]
        # Train es descriptivo; métricas de calibración inferenciales se revisan en held-out.
        for name in ['negativo','indeterminado','positivo']:
            m=result.categoria.to_numpy()==name;n=int(m.sum());k=int(truth[m].sum());lo,hi=wilson(k,n)
            summary.append({'conjunto':part,'categoria':name,'n':n,'positivos_reales':k,'negativos_reales':n-k,
                'P_roya_dado_categoria':k/n if n else None,'IC95_inferior':lo,'IC95_superior':hi})
        selected=result.positivo_RF.to_numpy();q=result.P_positivo_real_dado_RF_positivo.to_numpy()[selected]
        calibration.append({'conjunto':part,'Brier_global_RF':float(brier_score_loss(truth,result.probabilidad_RF)),
            'Brier_global_calibrado':float(brier_score_loss(truth,result.probabilidad_global_calibrada)),
            'Brier_condicional_red6':float(brier_score_loss(truth[selected],q))})
        for c in ['Municipio','Parcela','Planta','Muestreo']:result[c]=d.loc[mask,c].to_numpy()
        result['presencia_observada']=truth
        temp=OUT/('probabilidades_'+part+'.csv.tmp');result.to_csv(temp,index=False);temp.replace(OUT/('probabilidades_'+part+'.csv'))
    report={'umbral_negativo':low,'umbral_positivo':high,'umbrales_solo_pureza_agregada_no_adoptados':{'inferior':raw_low,'superior':raw_high},'metas_validacion':{'max_P_roya_en_negativos':.05,'min_P_roya_en_positivos':.90,
            'min_registros_por_categoria':200,'criterio':'Wilson 95% punto a punto en validación'},
        'resumen':summary,'calibracion':calibration,'calibrador_coef':calibrator.coef_.tolist(),'calibrador_intercept':calibrator.intercept_.tolist(),
        'semantica':'Probabilidad global calibrada con todos los scores OOF train; q condicional red6 sólo en positivos RF. No se multiplican.',
        'limites':['Intervalos Wilson nominales asumen ensayos independientes; registros repetidos por planta y búsqueda de umbral impiden interpretar cobertura 95% como garantía.',
                   'Metas de riesgo definidas como propuesta exploratoria, no costos clínicos/agrícolas acordados.',
                   'Prueba consultada repetidamente, mismos sitios/fechas; no validación prospectiva ni geográfica.',
                   'Ceros 17–24 pendientes de confirmar. Etiquetas reales significan etiquetas de la base.',
                   'Sin umbral suficiente la categoría queda vacía; indeterminado requiere revisión, no ausencia.',
                   'No modifica clasificador operativo ni entrena árbol difuso.']}
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));pd.DataFrame(summary).to_csv(OUT/'resumen_categorias.csv',index=False)
    joblib.dump(b,OUT/'modelo.joblib',compress=3);print(json.dumps(report,ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':run()
