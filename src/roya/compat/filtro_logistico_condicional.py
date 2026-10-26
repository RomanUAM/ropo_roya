"""RF positivo -> logística de falso/verdadero positivo -> cantidad GP.
La logística se entrena exclusivamente con positivos predichos OOF de train.
"""
import json,warnings
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,inputs,splitter,TARGET
from bosque_bayes_regresion import ROOT,enrich
from bosque_regresiones_gp import predict as base_predict

OUT=ROOT/'evaluacion_filtro_logistico'

def meta_features(f,p):
    a=f.copy();a['probabilidad_RF']=p
    p=np.clip(p,1e-6,1-1e-6);a['log_odds_RF']=np.log(p/(1-p))
    return a

def stats(y,result):
    yy=y>0;positive=result.presencia_predicha.to_numpy()
    tn=int((~yy&~positive).sum());fp=int((~yy&positive).sum());fn=int((yy&~positive).sum());tp=int((yy&positive).sum())
    return {'TN':tn,'FP':fp,'FN':fn,'TP':tp,'sensibilidad':tp/(tp+fn),
        'exactitud_balanceada':.5*(tp/(tp+fn)+tn/(tn+fp)),
        'tasa_falsos_positivos':fp/(fp+tn),'prob_alerta_falsa':fp/(fp+tp) if fp+tp else None,
        'MAE':float(mean_absolute_error(y,result.incidencia_predicha)),
        'RMSE':float(np.sqrt(mean_squared_error(y,result.incidencia_predicha)))}

def predict(b,d,variant=None):
    variant=b['selected_variant'] if variant is None else variant
    d=enrich(b['base'],d);base=base_predict(b['base'],d)
    first=base.presencia_predicha.to_numpy();p=base.probabilidad_presencia.to_numpy();q=np.full(len(d),np.nan)
    if variant=='referencia':
        keep=first.copy()
        # Informar la probabilidad condicional sin cambiar decisiones de referencia.
        if first.any():
            setting=b['variants']['balanceada'];model=b['logistic_models'][setting['C']]
            f=inputs(d,b['base']['simulation_bank'])
            q[first]=model.predict_proba(meta_features(f.iloc[np.flatnonzero(first)],p[first]))[:,1]
    else:
        setting=b['variants'][variant];model=b['logistic_models'][setting['C']]
        if first.any():
            f=inputs(d,b['base']['simulation_bank']);q[first]=model.predict_proba(meta_features(f.iloc[np.flatnonzero(first)],p[first]))[:,1]
        keep=first&(q>=setting['threshold'])
    quantity=np.where(keep,base.incidencia_predicha.to_numpy(),0.)
    return pd.DataFrame({'probabilidad_bosque':p,'positivo_bosque':first,
        'probabilidad_condicional_verdadero_positivo':q,
        'probabilidad_presencia':np.where(first,np.nan_to_num(q,nan=p),0.),
        'presencia_predicha':keep,'grupo_final':keep.astype(int),'incidencia_predicha':quantity,
        'estado':np.where(keep,'cantidad_GP',np.where(first,'rechazado_por_logistica','sin_roya_bosque'))})

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    def event(role,action,evidence):
        row={'fecha':datetime.now(timezone.utc).isoformat(),'agente':role,'accion':action,'evidencia':evidence}
        with (OUT/'eventos.jsonl').open('a') as s:s.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(action,str(evidence)[:650],flush=True)
    source=joblib.load(ROOT/'evaluacion_bosque_regresiones_gp/modelo.joblib')
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);pd.testing.assert_frame_equal(assignment,source['particion_plantas'])
    f=inputs(d,source['simulation_bank']);y=d[TARGET].to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba'
    ft=f[tr].reset_index(drop=True);yt=y[tr]>0;dt=d[tr].reset_index(drop=True)
    group=pd.MultiIndex.from_frame(dt[['Municipio','Parcela','Planta']]).factorize()[0]
    oof=np.full(len(ft),np.nan);folds=np.full(len(ft),-1,dtype=int);audit=[]
    for fold,(fit,held) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(ft,yt,group)):
        assert not set(group[fit])&set(group[held])
        prep=StandardScaler().fit(ft.iloc[fit]);model=clone(source['detector']).fit(prep.transform(ft.iloc[fit]),yt[fit])
        oof[held]=model.predict_proba(prep.transform(ft.iloc[held]))[:,1];folds[held]=fold
        row={'fold':fold,'train_registros':len(fit),'held_registros':len(held),
             'plantas_train':len(np.unique(group[fit])),'plantas_held':len(np.unique(group[held])),'interseccion_plantas':0}
        audit.append(row);event('datos','prediccion_fuera_de_muestra',row)
    assert np.isfinite(oof).all();selected=oof>=source['threshold'];target=yt[selected].astype(int)
    examples=dt[['Municipio','Parcela','Planta','Muestreo']].copy();examples['fold']=folds
    examples['probabilidad_RF_OOF']=oof;examples['positivo_RF_OOF']=selected;examples['presencia_real']=yt
    examples['etiqueta_meta']=np.where(selected,yt.astype(int),np.nan)
    examples['tipo']=np.where(selected,np.where(yt,'positivo_real_1','falso_positivo_0'),'no_entra_en_logistica')
    examples.to_csv(OUT/'predicciones_train_OOF.csv',index=False)
    train_meta=meta_features(ft[selected],oof[selected])
    base_v=base_predict(source,d[va]);first_v=base_v.presencia_predicha.to_numpy();pv=base_v.probabilidad_presencia.to_numpy()
    vm=meta_features(f[va].iloc[np.flatnonzero(first_v)],pv[first_v]);truth=(y[va]>0)[first_v]
    bundle={'base':source,'logistic_models':{},'variants':{},'selected_variant':'referencia','schema':'logistica_condicional_v1'}
    base_stats=stats(y[va],base_v);conservative=[];balanced=[];model_rows=[]
    for C in [.01,.1,1.,10.]:
        name=str(C);model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=42))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(train_meta,target)
        notices=[str(w.message) for w in caught];bundle['logistic_models'][name]=model
        q=model.predict_proba(vm)[:,1];threshold=float(q[truth].min())
        def score(t):
            keep=first_v.copy();keep[first_v]=q>=t
            rr=base_v.copy();rr['presencia_predicha']=keep;rr['incidencia_predicha']=np.where(keep,rr.incidencia_predicha,0.)
            return stats(y[va],rr)
        conservative.append({'C':name,'threshold':threshold,'validacion':score(threshold),'advertencias':notices})
        candidates=[{'C':name,'threshold':float(t),'validacion':score(t),'advertencias':notices} for t in np.linspace(0,1,201)]
        balanced.append(max(candidates,key=lambda r:r['validacion']['exactitud_balanceada']))
        row={'C':C,'advertencias':notices,'n_train_meta':len(target),
             'Brier_condicional_validacion':float(brier_score_loss(truth,q)),
             'AUC_condicional_validacion':float(roc_auc_score(truth,q)),
             'Brier_RF_en_positivos_validacion':float(brier_score_loss(truth,pv[first_v]))}
        model_rows.append(row);event('modelacion','logistica_condicional',row)
    allowed=[r for r in conservative if not r['advertencias']]
    cons=min(allowed,key=lambda r:(r['validacion']['FP'],r['validacion']['MAE']))
    bal=max([r for r in balanced if not r['advertencias']],key=lambda r:r['validacion']['exactitud_balanceada'])
    bundle['variants']={'conservadora':cons,'balanceada':bal}
    improve=cons['validacion']['FP']<base_stats['FP'] and cons['validacion']['FN']<=base_stats['FN']
    bundle['selected_variant']='conservadora' if improve else 'referencia'
    validation=[{'variante':'referencia',**base_stats}]+[{'variante':n,**setting['validacion']} for n,setting in bundle['variants'].items()]
    testing=[]
    for variant in ('referencia','conservadora','balanceada'):
        result=predict(bundle,d[te],variant);testing.append({'variante':variant,**stats(y[te],result)})
        result['incidencia_observada']=y[te]
        for c in ['Municipio','Parcela','Planta','Muestreo']:result[c]=d.loc[te,c].to_numpy()
        result.to_csv(OUT/('predicciones_'+variant+'.csv'),index=False)
    report={'diseno':'50/20/30 por plantas; cinco pliegues agrupados sólo dentro de train para etiquetas meta.',
        'entrenamiento_meta':{'positivos_predichos_OOF':int(selected.sum()),'falsos_positivos_etiqueta_0':int((target==0).sum()),
                             'positivos_reales_etiqueta_1':int((target==1).sum()),'pliegues':audit},
        'umbral_RF':source['threshold'],'features_meta':list(train_meta),'modelos_logisticos':model_rows,
        'configuraciones':bundle['variants'],'comparacion_validacion':validation,'comparacion_prueba':testing,
        'seleccion_validacion':bundle['selected_variant'],'mejora_conservadora_validacion':bool(improve),
        'prueba':next(r for r in testing if r['variante']==bundle['selected_variant']),
        'semantica':'q=P(roya real | RF predice positivo, X, score RF). No multiplicar automáticamente por p_RF; no es probabilidad global calibrada.',
        'limites':['La segunda etapa sólo rechaza positivos del bosque; no recupera falsos negativos iniciales.',
                   'No se entrena con falsos positivos de validación o prueba; sólo OOF de entrenamiento.',
                   'Mismas parcelas y fechas: no generalización temporal/geográfica; prueba previamente consultada.',
                   'Muestreos 17–24 totalmente cero, pendiente confirmar significado; no se resolvió el problema de datos.',
                   'La variante balanceada puede aumentar omisiones; no se promueve automáticamente.',
                   'Probabilidad condicional debe comprobarse con cohorte independiente; Brier sólo en seleccionados.',
                   'No se reabre la comparación del árbol difuso ni se sustituye la evaluación temporal anterior.']}
    first_t=result.positivo_bosque.to_numpy();pt=result.probabilidad_bosque.to_numpy()
    testmeta=meta_features(f[te].iloc[np.flatnonzero(first_t)],pt[first_t]);truth_t=(y[te]>0)[first_t]
    calibration=[];bins=[]
    for C,model in bundle['logistic_models'].items():
        qt=model.predict_proba(testmeta)[:,1]
        calibration.append({'C':C,'Brier_condicional_prueba':float(brier_score_loss(truth_t,qt)),
                            'AUC_condicional_prueba':float(roc_auc_score(truth_t,qt))})
        for j in range(10):
            mask=(qt>=j/10)&(qt<((j+1)/10 if j<9 else 1.00001))
            bins.append({'C':C,'intervalo_inferior':j/10,'n':int(mask.sum()),
                         'prob_media':float(qt[mask].mean()) if mask.any() else None,
                         'frecuencia_real':float(truth_t[mask].mean()) if mask.any() else None})
    report['calibracion_condicional_prueba']=calibration
    report['Brier_RF_condicionado_prueba']=float(brier_score_loss(truth_t,pt[first_t]))
    pd.DataFrame(bins).to_csv(OUT/'calibracion_condicional_prueba.csv',index=False)
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    joblib.dump(bundle,OUT/'modelo.joblib',compress=3)
    pd.DataFrame(validation).to_csv(OUT/'comparacion_validacion.csv',index=False)
    pd.DataFrame(testing).to_csv(OUT/'comparacion_prueba.csv',index=False)
    event('auditoria','seleccion_validacion',{'elegida':bundle['selected_variant'],'validacion':validation,'prueba':testing})
    return report

if __name__=='__main__':run()
