"""Seis detectores, dos confirmadores OOF y trece regresiones con conteo 15d."""
import json,warnings
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,splitter,quantity_model,quantity_predict,TARGET
from bosque_bayes_regresion import ROOT,enrich,NAMES
from programacion_genetica_roya import SymbolicRegressor3
from probar_arbitro_logistico import cycle
from ventana15_roya import build_bank,features
from filtro_logistico_condicional import meta_features

OUT=ROOT/'evaluacion_15dias'
PRIMARY=['bosque','logistica','svm_lineal','red_6_9_12','red_6_9_18_36','red_6_9_18_36_36']
REGRESSORS=NAMES+['genetica_3_niveles_s'+str(i) for i in (42,43,44)]

def save_csv(frame,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temporary,index=False)
    temporary.replace(path)

def preprocess():return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler())

def classifier(name,y,groups,C=1):
    if name=='bosque':return RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=10,n_jobs=2,random_state=42)
    if name=='logistica':return LogisticRegression(C=C,max_iter=2000,random_state=42)
    if name=='svm_lineal':
        cv=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(np.zeros((len(y),1)),y,groups))
        return CalibratedClassifierCV(LinearSVC(C=C,dual=False,max_iter=10000,random_state=42),method='sigmoid',cv=cv,n_jobs=1)
    return MLPClassifier(hidden_layer_sizes=tuple(int(v) for v in name[4:].split('_')),alpha=1,max_iter=1000,early_stopping=False,random_state=42)

def binary_stats(y,keep):
    yy=y>0;tp=int((yy&keep).sum());fn=int((yy&~keep).sum());fp=int((~yy&keep).sum());tn=int((~yy&~keep).sum())
    return {'TN':tn,'FP':fp,'FN':fn,'TP':tp,'sensibilidad':tp/(tp+fn),
            'exactitud_balanceada':.5*(tp/(tp+fn)+tn/(tn+fp)),
            'prob_alerta_falsa':fp/(fp+tp) if fp+tp else None}

def quantity(model,name,z,d):
    a=np.column_stack([z,cycle(d)]) if name=='ciclico' else z
    return quantity_predict(model,name,a)

def predict(b,d,primary=None,route=None,regression=None):
    primary=b['selected_primary'] if primary is None else primary
    route=(b['selected_route'] if primary==b['selected_primary'] else 'sin_confirmacion') if route is None else route
    if primary!=b['selected_primary'] and route!='sin_confirmacion':raise ValueError('Confirmadores entrenados sólo para el detector elegido.')
    regression=b['selected_regression'] if regression is None else regression
    d=enrich(b,d);f=features(d,b['bank']);x=b['preprocessor'].transform(f)
    p=b['primary_models'][primary].predict_proba(x)[:,1];first=p>=b['primary_thresholds'][primary]
    q=np.full(len(d),np.nan);keep=first.copy()
    if route!='sin_confirmacion' and first.any():
        config=b['routes'][route];model=b['confirmers'][config['model_key']]
        q[first]=model.predict_proba(meta_features(f.iloc[np.flatnonzero(first)],p[first]))[:,1]
        keep=first&(q>=config['threshold'])
    amount=np.zeros(len(d))
    if keep.any():
        idx=np.flatnonzero(keep);z=b['positive_preprocessor'].transform(f.iloc[idx])
        amount[keep]=quantity(b['regressors'][regression],regression,z,d.iloc[idx])
    return pd.DataFrame({'probabilidad_detector':p,'positivo_detector':first,
        'probabilidad_condicional_confirmador':q,'probabilidad_presencia':np.where(first,np.nan_to_num(q,nan=p),0.),
        'presencia_predicha':keep,'grupo_final':keep.astype(int),'incidencia_predicha':amount,
        'estado':np.where(keep,'cantidad_estimada',np.where(first,'rechazado_por_confirmador','sin_roya_detector'))})

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    def event(role,action,evidence):
        row={'fecha':datetime.now(timezone.utc).isoformat(),'agente':role,'accion':action,'evidencia':evidence}
        with (OUT/'eventos.jsonl').open('a') as s:s.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(action,str(evidence)[:550],flush=True)
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);y=d[TARGET].to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';pos=tr&(y>0)
    ids=pd.MultiIndex.from_frame(d[['Municipio','Parcela','Planta']]).factorize()[0];groups=ids[tr]
    bank=build_bank(ROOT/'simulacion_diaria_original');f=features(d,bank)
    f[bank['columns']].to_csv(OUT/'conteos15_por_registro.csv',index=False)
    assignment.to_csv(OUT/'particion_plantas.csv',index=False)
    prep=preprocess();x=prep.fit_transform(f[tr]);v=prep.transform(f[va]);primary_models={};thresholds={};primary_rows=[]
    for name in PRIMARY:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model=classifier(name,y[tr]>0,groups).fit(x,y[tr]>0)
        p=model.predict_proba(v)[:,1];grid=np.arange(.05,.951,.01)
        threshold=float(grid[np.argmax([balanced_accuracy_score(y[va]>0,p>=t) for t in grid])])
        row={'modelo':name,'threshold':threshold,'advertencias':[str(w.message) for w in caught],
             'validacion':binary_stats(y[va],p>=threshold)}
        primary_rows.append(row);primary_models[name]=model;thresholds[name]=threshold;event('modelacion','detector_15dias',row)
    eligible=[r for r in primary_rows if not r['advertencias']]
    best=max(eligible,key=lambda r:r['validacion']['exactitud_balanceada'])['modelo']
    pval=primary_models[best].predict_proba(v)[:,1];first=pval>=thresholds[best];baseval=binary_stats(y[va],first)
    ft=f[tr].reset_index(drop=True);yt=y[tr]>0;oof=np.full(tr.sum(),np.nan);folds=np.full(tr.sum(),-1,int)
    for fold,(fit,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ft,yt,groups)):
        assert not set(groups[fit])&set(groups[held])
        pp=preprocess();xx=pp.fit_transform(ft.iloc[fit]);est=classifier(best,yt[fit],groups[fit]).fit(xx,yt[fit])
        oof[held]=est.predict_proba(pp.transform(ft.iloc[held]))[:,1];folds[held]=fold
        event('datos','OOF_detector_elegido',{'fold':fold,'modelo':best,'held_registros':len(held),'plantas_compartidas':0})
    selected=oof>=thresholds[best];meta_y=yt[selected];meta_groups=groups[selected]
    examples=d.loc[tr,['Municipio','Parcela','Planta','Muestreo']].reset_index(drop=True)
    examples['fold']=folds;examples['prob_detector_OOF']=oof;examples['positivo_detector_OOF']=selected
    examples['etiqueta_confirmador']=np.where(selected,yt.astype(int),np.nan);examples.to_csv(OUT/'ejemplos_confirmadores_OOF.csv',index=False)
    meta=meta_features(ft[selected],oof[selected]);mv=meta_features(f[va].iloc[np.flatnonzero(first)],pval[first]);truth=(y[va]>0)[first]
    confirmers={};route_rows=[];routes={}
    for name in ['logistica','svm_lineal']:
        for C in [.1,1.,10.]:
            key=name+'_C'+str(C)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),classifier(name,meta_y,meta_groups,C)).fit(meta,meta_y)
            notices=[str(w.message) for w in caught];confirmers[key]=model;q=model.predict_proba(mv)[:,1]
            def score(t):
                keep=first.copy();keep[first]=q>=t;return binary_stats(y[va],keep)
            conservative=float(q[truth].min());grid=np.linspace(0,1,201)
            balanced=max(grid,key=lambda t:score(t)['exactitud_balanceada'])
            for mode,t in [('conservadora',conservative),('balanceada',float(balanced))]:
                route_rows.append({'tipo':name,'modo':mode,'model_key':key,'threshold':t,
                                   'advertencias':notices,'validacion':score(t)})
            event('modelacion','confirmador_15dias',{'modelo':key,'FP_meta':int((~meta_y).sum()),'TP_meta':int(meta_y.sum()),'advertencias':notices})
    for name in ['logistica','svm_lineal']:
        for mode in ['conservadora','balanceada']:
            options=[r for r in route_rows if r['tipo']==name and r['modo']==mode and not r['advertencias']]
            if not options:continue
            chosen=(min(options,key=lambda r:r['validacion']['FP']) if mode=='conservadora' else max(options,key=lambda r:r['validacion']['exactitud_balanceada']))
            routes[name+'_'+mode]=chosen
    candidates=[(name,c) for name,c in routes.items() if name.endswith('conservadora') and c['validacion']['FP']<baseval['FP'] and c['validacion']['FN']<=baseval['FN']]
    chosen_route=min(candidates,key=lambda item:item[1]['validacion']['FP'])[0] if candidates else 'sin_confirmacion'
    b={'bank':bank,'preprocessor':prep,'primary_models':primary_models,'primary_thresholds':thresholds,
       'selected_primary':best,'confirmers':confirmers,'routes':routes,'selected_route':chosen_route,
       'regressors':{},'selected_regression':'','lookup_geografia':d[['Municipio','Parcela','Coordenada_X','Coordenada_Y']].copy(),
       'particion_plantas':assignment,'schema':'ventana15_v1'}
    # Geografía completa para inferir desde el formato original.
    previous=joblib.load(ROOT/'evaluacion_bosque_regresiones_gp/modelo.joblib');b['lookup_geografia']=previous['lookup_geografia']
    pp_positive=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),PCA(n_components=.95,svd_solver='full'))
    a=pp_positive.fit_transform(f[pos]);zv=pp_positive.transform(f[va]);b['positive_preprocessor']=pp_positive
    keep=first.copy()
    if chosen_route!='sin_confirmacion':
        config=routes[chosen_route];q=confirmers[config['model_key']].predict_proba(mv)[:,1];keep[first]=q>=config['threshold']
    regression_rows=[]
    for name in REGRESSORS:
        model=SymbolicRegressor3(seed=int(name.rsplit('s',1)[1])) if name.startswith('genetica_') else quantity_model(name)
        xx=np.column_stack([a,cycle(d[pos])]) if name=='ciclico' else a
        target=np.log1p(y[pos]) if name=='log1p_lineal' else y[pos]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(xx,target)
        b['regressors'][name]=model;amount=quantity(model,name,zv,d[va]);final=np.where(keep,amount,0.)
        positive_val=y[va]>0
        row={'modelo':name,'advertencias':[str(w.message) for w in caught],
             'MAE_validacion':float(mean_absolute_error(y[va],final)),
             'RMSE_validacion':float(np.sqrt(mean_squared_error(y[va],final))),
             'MAE_positivos_validacion':float(mean_absolute_error(y[va][positive_val],amount[positive_val]))}
        regression_rows.append(row);event('modelacion','regresor_15dias',row)
    b['selected_regression']=min([r for r in regression_rows if not r['advertencias']],key=lambda r:r['MAE_validacion'])['modelo']
    primary_test=[];confirm_test=[];regression_test=[]
    testx=prep.transform(f[te])
    for name in PRIMARY:
        p=primary_models[name].predict_proba(testx)[:,1];primary_test.append({'modelo':name,**binary_stats(y[te],p>=thresholds[name])})
    for route in ['sin_confirmacion']+list(routes):
        result=predict(b,d[te],route=route)
        confirm_test.append({'ruta':route,**binary_stats(y[te],result.presencia_predicha.to_numpy()),
             'MAE':float(mean_absolute_error(y[te],result.incidencia_predicha))})
        result['incidencia_observada']=y[te];save_csv(result,OUT/('predicciones_ruta_'+route+'.csv'))
    for name in REGRESSORS:
        result=predict(b,d[te],regression=name)
        regression_test.append({'modelo':name,'MAE':float(mean_absolute_error(y[te],result.incidencia_predicha)),
                               'RMSE':float(np.sqrt(mean_squared_error(y[te],result.incidencia_predicha)))})
        result['incidencia_observada']=y[te];save_csv(result,OUT/('predicciones_regresion_'+name+'.csv'))
    report={'diseno':'50/20/30 por plantas completas, semilla 42; mismas parcelas y fechas conocidas.',
        'ventana':15,'regla':'Suma de 15 indicadores diarios: respuesta térmica de realización 1 >=0.9. Germinación y apresorio por separado. Hoy excluido; no promedio de días ni fracciones de escenarios.',
        'fecha':'Primer día posible de quincena (1/16), o Fecha_muestreo_exacta si se aporta. Primeras ventanas incompletas NaN; sin circularidad.',
        'predictores':list(f),'dimensiones_transformadas':int(x.shape[1]),'componentes_positivos':int(a.shape[1]),
        'detectores_validacion':primary_rows,'detectores_prueba':primary_test,'detector_elegido':best,
        'confirmadores_busqueda':route_rows,'rutas_elegidas':routes,'ruta_elegida':chosen_route,
        'confirmadores_prueba':confirm_test,'meta_train':{'n':int(selected.sum()),'FP_0':int((~meta_y).sum()),'TP_1':int(meta_y.sum())},
        'regresiones_validacion':regression_rows,'regresiones_prueba':regression_test,'regresion_elegida':b['selected_regression'],
        'limites':['Respuesta>=0.9 es umbral térmico exploratorio guardado, no infección confirmada; no agrega reglas de mojado foliar inexistentes.',
                   'Se cambia también la representación de las ventanas; no se atribuye diferencia exclusivamente a pasar de 7 a 15.',
                   'Prueba previamente consultada; no evaluación temporal/geográfica. Persisten ceros de muestreos 17–24 sin confirmar.',
                   'Confirmadores sólo rechazan positivos: pueden aumentar FN y no recuperan omisiones iniciales.',
                   'No se entrena árbol difuso; su condición previa continúa sin satisfacerse.']}
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));joblib.dump(b,OUT/'modelo.joblib',compress=3)
    pd.DataFrame(primary_test).to_csv(OUT/'detectores_prueba.csv',index=False)
    pd.DataFrame(confirm_test).to_csv(OUT/'confirmadores_prueba.csv',index=False)
    pd.DataFrame(regression_test).to_csv(OUT/'regresiones_prueba.csv',index=False)
    event('auditoria','seleccion_final',{'detector':best,'ruta':chosen_route,'regresion':b['selected_regression']})
    return report

if __name__=='__main__':run()
