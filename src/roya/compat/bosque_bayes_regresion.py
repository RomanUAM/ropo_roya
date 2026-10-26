"""Bosque binario -> clases latentes bayesianas / red sobre ACP -> regresión.
Modelo bayesiano: grupo latente -> componentes gaussianos condicionalmente
independientes. No aprende un DAG causal general ni clases biológicas.
"""
import json,warnings,hashlib
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,balanced_accuracy_score,accuracy_score
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import (assemble,inputs,splitter,pp,quantity_model,
                                quantity_predict,binary_metrics,GEO,TERRAIN,TARGET)
from probar_arbitro_logistico import cycle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evaluacion_bosque_bayes'
NAMES=['lineal','log1p_lineal','polinomio_2','polinomio_3','ciclico',
       'red_10_10_10','red_6_9_12','red_6_9_18_36','red_6_9_18_36_36','mediana_grupo']

def normalize(p):
    return p/np.maximum(p.sum(axis=1,keepdims=True),1e-30)

def bayes_proba(b,z):
    return normalize(b['mixture'].predict_proba(z)[:,b['active']])

def enrich(b,d):
    needed=GEO+TERRAIN+['Orientacion_ladera_MDE_grados']
    missing=[c for c in needed if c not in d]
    if not missing:return d
    left=d.copy();right=b['lookup_geografia'].copy()
    for data in (left,right):
        data['_x']=data.Coordenada_X.round(6);data['_y']=data.Coordenada_Y.round(6)
    keys=['Municipio','Parcela','_x','_y']
    result=left.merge(right[keys+missing].drop_duplicates(keys),on=keys,how='left',validate='many_to_one',sort=False).drop(columns=['_x','_y'])
    if result[missing].isna().any().any():raise ValueError('Sitio sin geografía: no se imputa cero.')
    return result

def reg_predict(model,name,z,d):
    a=np.column_stack([z,cycle(d)]) if name=='ciclico' else z
    return quantity_predict(model,name,a)

def predict(b,d,variant=None):
    variant=b['selected_variant'] if variant is None else variant
    d=enrich(b,d);f=inputs(d,b['simulation_bank'])
    p=b['detector'].predict_proba(b['presence_pp'].transform(f))[:,1]
    present=p>=b['threshold'];z=b['positive_pp'].transform(f)
    group=np.zeros(len(d),dtype=int);q=np.zeros(len(d));certainty=np.full(len(d),np.nan)
    if present.any():
        dz=d.iloc[np.flatnonzero(present)];zz=z[present]
        if variant=='global':
            model,name=b['global_model'];q[present]=reg_predict(model,name,zz,dz)
            group[present]=1
        else:
            source,mode=variant.rsplit(':',1)
            probs=bayes_proba(b,zz) if source=='bayes' else b['networks'][source].predict_proba(zz)
            g=probs.argmax(axis=1);group[present]=g+1;certainty[present]=probs.max(axis=1)
            if mode=='soft':
                estimates=np.column_stack([reg_predict(model,name,zz,dz) for model,name in b['regressions']])
                q[present]=(probs*estimates).sum(axis=1)
            else:
                values=np.empty(len(zz))
                for j,(model,name) in enumerate(b['regressions']):
                    mask=g==j
                    if mask.any():values[mask]=reg_predict(model,name,zz[mask],dz.iloc[np.flatnonzero(mask)])
                q[present]=values
    return pd.DataFrame({'probabilidad_presencia':p,'presencia_predicha':present,
        'grupo_final':group,'probabilidad_grupo_maxima':certainty,'incidencia_predicha':q,
        'estado':np.where(present,'cantidad_estimada','sin_roya')})

def metrics(y,result,threshold):
    return {**binary_metrics(y>0,result.probabilidad_presencia.to_numpy(),threshold),
            'MAE_cantidad':float(mean_absolute_error(y,result.incidencia_predicha)),
            'RMSE_cantidad':float(np.sqrt(mean_squared_error(y,result.incidencia_predicha)))}

def run():
    threadpool_limits(2);out=OUT;out.mkdir(exist_ok=False)
    def event(role,action,evidence):
        row={'fecha':datetime.now(timezone.utc).isoformat(),'agente':role,'accion':action,'evidencia':evidence}
        with (out/'eventos.jsonl').open('a') as s:s.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(action,str(evidence)[:550],flush=True)
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    prior=joblib.load(ROOT/'evaluacion_conteos_completas/modelo.joblib')
    bank=prior['simulation_bank'];f=inputs(d,bank);y=d[TARGET].to_numpy();split,assignment=splitter(d)
    pd.testing.assert_frame_equal(assignment,prior['particion_plantas'])
    assignment.to_csv(out/'particion_plantas.csv',index=False)
    tr=split=='train';va=split=='validacion';te=split=='prueba';pos=tr&(y>0);vp=va&(y>0)
    prep=make_pipeline(StandardScaler());x=prep.fit_transform(f[tr]);v=prep.transform(f[va])
    forest_results=[];forests=[]
    for depth,leaf in [(12,20),(12,10),(None,20),(None,10)]:
        model=RandomForestClassifier(n_estimators=200,max_depth=depth,min_samples_leaf=leaf,n_jobs=2,random_state=42).fit(x,y[tr]>0)
        prob=model.predict_proba(v)[:,1];thresholds=np.arange(.05,.951,.01)
        threshold=float(thresholds[np.argmax([balanced_accuracy_score(y[va]>0,prob>=t) for t in thresholds])])
        row={'profundidad':depth,'hojas_minimas':leaf,'umbral':threshold,'validacion':binary_metrics(y[va]>0,prob,threshold)}
        forest_results.append(row);forests.append(model);event('modelacion','bosque_reentrenado',row)
    chosen=max(range(len(forest_results)),key=lambda i:forest_results[i]['validacion']['exactitud_balanceada'])
    detector=forests[chosen];threshold=forest_results[chosen]['umbral']
    positive_pp=pp();a=positive_pp.fit_transform(f[pos]);zv=positive_pp.transform(f[va]);av=positive_pp.transform(f[vp])
    mixtures=[];mixture_rows=[]
    for cap in [1,2,3,4,6,8,12]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            m=BayesianGaussianMixture(n_components=cap,covariance_type='diag',weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=.1,reg_covar=.01,n_init=2,max_iter=1000,random_state=42).fit(a)
        active=np.flatnonzero(np.bincount(m.predict(a),minlength=cap)>=100)
        notices=[str(w.message) for w in caught]
        valid_counts=np.bincount(m.predict(av),minlength=cap)
        eligible=bool(m.converged_ and len(active)==cap and (valid_counts>=30).all())
        row={'limite_grupos':cap,'grupos_con_100_positivos_train':len(active),
             'log_densidad_validacion':float(m.score(av)),'convergio':bool(m.converged_),
             'advertencias':notices,'elegible':eligible,'pesos':m.weights_.tolist(),
             'train_por_grupo':np.bincount(m.predict(a),minlength=cap).tolist(),'validacion_por_grupo':valid_counts.tolist()}
        mixtures.append(m);mixture_rows.append(row);event('estadistica','clases_latentes_bayesianas',row)
    eligible=[i for i,r in enumerate(mixture_rows) if r['elegible']]
    if not eligible:raise RuntimeError('No hay agrupación bayesiana con soporte suficiente.')
    mi=max(eligible,key=lambda i:mixture_rows[i]['log_densidad_validacion']);mixture=mixtures[mi]
    active=np.arange(mixture.n_components);g=mixture.predict(a);gv=mixture.predict(av);k=len(active)
    event('estadistica','grupos_seleccionados',mixture_rows[mi])
    networks={};network_results={}
    if k>1:
        for layers in [(6,9,18,36),(6,9,18,36,36)]:
            name='red_'+'_'.join(map(str,layers))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model=MLPClassifier(hidden_layer_sizes=layers,alpha=1,max_iter=1000,early_stopping=False,random_state=42).fit(a,g)
            notices=[str(w.message) for w in caught]
            network_results[name]={'acierto_etiquetas_bayesianas_validacion':float(accuracy_score(gv,model.predict(av))),
                                   'advertencias':notices,'iteraciones':int(model.n_iter_)}
            if not notices:networks[name]=model
            event('modelacion','red_para_grupos',network_results[name])
    def regress(fit,valid,label):
        fitted={};scores={};notices={}
        for name in NAMES:
            aa=np.column_stack([a[fit],cycle(d[pos])[fit]]) if name=='ciclico' else a[fit]
            vv=np.column_stack([zv[valid],cycle(d[va])[valid]]) if name=='ciclico' else zv[valid]
            yy=np.log1p(y[pos][fit]) if name=='log1p_lineal' else y[pos][fit]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model=quantity_model(name).fit(aa,yy)
            notices[name]=[str(w.message) for w in caught]
            scores[name]=float(mean_absolute_error(y[va][valid],quantity_predict(model,name,vv)))
            fitted[name]=model
        allowed=[n for n in NAMES if not notices[n]]
        if not allowed:raise RuntimeError('Regresiones sin convergencia: '+label)
        selected=min(allowed,key=lambda n:scores[n]);row={'grupo':label,'positivos_train':int(fit.sum()),
             'positivos_validacion':int(valid.sum()),'MAE_validacion':scores,'advertencias':notices,'elegido':selected}
        event('modelacion','regresion_seleccionada',row)
        return (fitted[selected],selected),row
    regressions=[];reg_reports=[];gall=mixture.predict(zv)
    for j in range(k):
        selected,row=regress(g==j,(gall==j)&(y[va]>0),j+1);regressions.append(selected);reg_reports.append(row)
    global_model,global_report=regress(np.ones(len(a),dtype=bool),y[va]>0,'global_sin_grupos')
    bundle={'simulation_bank':bank,'feature_columns':list(f),'presence_pp':prep,'positive_pp':positive_pp,
            'detector':detector,'threshold':threshold,'mixture':mixture,'active':active,'networks':networks,
            'regressions':regressions,'global_model':global_model,'lookup_geografia':prior['lookup_geografia'],
            'particion_plantas':assignment,'selected_variant':'bayes:hard','schema':'bosque_bayes_v1'}
    variants=['bayes:hard','bayes:soft']+[n+':'+mode for n in networks for mode in ['hard','soft']]+['global']
    validation=[]
    for variant in variants:
        result=predict(bundle,d[va],variant)
        validation.append({'variante':variant,**metrics(y[va],result,threshold)})
    selected=min(validation,key=lambda r:r['MAE_cantidad'])['variante']
    selected_group=min([r for r in validation if r['variante']!='global'],key=lambda r:r['MAE_cantidad'])['variante']
    neural_rows=[r for r in validation if r['variante'].startswith('red_')]
    selected_neural=min(neural_rows,key=lambda r:r['MAE_cantidad'])['variante'] if neural_rows else None
    bundle.update(selected_variant=selected,selected_group_variant=selected_group,selected_neural_variant=selected_neural)
    event('auditoria','seleccion_solo_validacion',{'elegido':selected,'comparacion':validation})
    testing=[]
    for variant in variants:
        result=predict(bundle,d[te],variant);testing.append({'variante':variant,**metrics(y[te],result,threshold)})
        result['incidencia_observada']=y[te]
        for col in ['Municipio','Parcela','Planta','Mes_registrado','Muestreo']:result[col]=d.loc[te,col].to_numpy()
        result.to_csv(out/('predicciones_'+variant.replace(':','_')+'.csv'),index=False)
    report={'detector_elegido':'bosque_reentrenado','umbral':threshold,'bosques':forest_results,'bosque_elegido':forest_results[chosen],
            'diseno':'50/20/30 por plantas completas, mismas parcelas y fechas; positivos heredan partición',
            'features':list(f),'componentes_presencia':f.shape[1],'componentes_positivos':a.shape[1],
            'busqueda_grupos':mixture_rows,'numero_grupos':k,'elegido_grupos':mixture_rows[mi],
            'criterio_grupos':'Mayor log densidad de positivos de validación entre límites elegibles 1,2,3,4,6,8,12; soporte mínimo 100 train y 30 validación por grupo.',
            'redes_grupos':network_results,'grupos':reg_reports,'regresion_global':global_report,
            'comparacion_validacion':validation,'seleccion_validacion':selected,'seleccion_con_grupos':selected_group,'seleccion_neuronal':selected_neural,'comparacion_prueba':testing,
            'prueba':next(r for r in testing if r['variante']==selected),'modelo_bayesiano':'Grupo latente -> PCs; mezcla gaussiana diagonal con prior de proceso de Dirichlet truncado. No aprende DAG general ni causalidad.',
            'limites':['Prueba previamente consultada: evaluación exploratoria.',
                       'Grupos son estadísticos, no clases biológicas; límite superior alcanzado no prueba número natural de grupos.',
                       'ACP no implica independencia condicional dentro de los grupos; aproximación diagonal por evaluar.',
                       'Fecha exacta desconocida: ventanas promediadas por quincena.',
                       'Clima siguiente es escenario anual circular, no pronóstico validado.',
                       'No evalúa nuevas parcelas ni periodos futuros; CRS y altitudes pendientes.'],
            'fuentes_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv']}}
    (out/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));joblib.dump(bundle,out/'modelo.joblib',compress=3)
    pd.DataFrame(validation).to_csv(out/'comparacion_validacion.csv',index=False);pd.DataFrame(testing).to_csv(out/'comparacion_prueba.csv',index=False)
    event('auditoria','resultado_final',report['prueba'])
    return report

if __name__=='__main__':run()
