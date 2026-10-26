"""Mejora por antecedentes y mezcla GP/log; avance difuso condicionado a validación."""
import json,warnings
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,TARGET,quantity_model,quantity_predict
from bosque_bayes_regresion import ROOT,enrich
from rasgos_temporales_roya import features,history_columns,climate_bank
from programacion_genetica_roya import SymbolicRegressor3

OUT=ROOT/'evaluacion_mejora_temporal'

def statistics(y,p,q,threshold):
    yy=y>0;pred=p>=threshold
    tp=int((yy&pred).sum());fn=int((yy&~pred).sum());fp=int((~yy&pred).sum());tn=int((~yy&~pred).sum())
    sensitivity=tp/(tp+fn) if tp+fn else None;spec=tn/(tn+fp) if tn+fp else None
    return {'TN':tn,'FP':fp,'FN':fn,'TP':tp,'sensibilidad':sensitivity,
            'exactitud_balanceada':(sensitivity+spec)/2 if sensitivity is not None and spec is not None else None,
            'tasa_falsos_positivos':fp/(fp+tn) if fp+tn else None,
            'prob_alerta_falsa':fp/(fp+tp) if fp+tp else None,
            'MAE':float(mean_absolute_error(y,q)),'RMSE':float(np.sqrt(mean_squared_error(y,q)))}

def recall_threshold(y,p,recall=.98):
    values=np.sort(p[y>0]);required=int(np.ceil(recall*len(values)))
    if not len(values):raise ValueError('Validación sin positivos: no se ajusta sensibilidad.')
    return float(values[len(values)-required])

def split_designs(d):
    periods=d.Muestreo.astype(str).str.extract(r'(\d+)')[0].astype(int).to_numpy()
    temporal=np.where(periods<=12,'train',np.where(periods<=17,'validacion','prueba'))
    assignment={};rng=np.random.default_rng(42)
    for municipality,g in d[['Municipio','Parcela']].drop_duplicates().groupby('Municipio',sort=True):
        plots=sorted(g.Parcela.tolist());order=rng.permutation(len(plots))
        for rank,i in enumerate(order):assignment[(municipality,plots[i])]='train' if rank<3 else ('validacion' if rank==3 else 'prueba')
    spatial=np.array([assignment[(m,p)] for m,p in zip(d.Municipio,d.Parcela)])
    return {'temporal_12_5_7':temporal,'parcelas_6_2_4':spatial}

def predict(b,d):
    d=enrich(b,d);f=features(d,b.get('climate_bank'),b['enhanced'])
    p=b['forest'].predict_proba(b['presence_pp'].transform(f))[:,1]
    active=p>=b['threshold'];q=np.zeros(len(d))
    if active.any():
        z=b['positive_pp'].transform(f.iloc[np.flatnonzero(active)])
        a=b['gp'].predict(z);c=quantity_predict(b['log'],'log1p_lineal',z)
        q[active]=b['gp_weight']*a+(1-b['gp_weight'])*c
    return pd.DataFrame({'probabilidad_presencia':p,'presencia_predicha':active,
        'incidencia_predicha':q,'grupo_final':active.astype(int),
        'estado':np.where(active,'mezcla_genetica_logaritmica','sin_roya')})

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    def event(role,action,evidence):
        row={'fecha':datetime.now(timezone.utc).isoformat(),'agente':role,'accion':action,'evidencia':evidence}
        with (OUT/'eventos.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(action,str(evidence)[:650],flush=True)
    raw=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    d=history_columns(raw);y=d[TARGET].to_numpy();period=d.Muestreo.astype(str).str.extract(r'(\d+)')[0].astype(int)
    audit=d.assign(periodo=period).groupby('periodo')[TARGET].agg(registros='size',positivos=lambda x:int((x>0).sum()),media='mean').reset_index()
    audit.to_csv(OUT/'auditoria_objetivo_por_muestreo.csv',index=False)
    event('datos','auditoria_ceros',audit.to_dict(orient='records'))
    bank=climate_bank();source=joblib.load(ROOT/'evaluacion_bosque_bayes/modelo.joblib');designs=split_designs(d)
    reports={};validation_gate=[]
    for design,split in designs.items():
        out=OUT/design;out.mkdir();tr=split=='train';va=split=='validacion';te=split=='prueba';positive=tr&(y>0)
        summary={s:{'n':int((split==s).sum()),'positivos':int(((split==s)&(y>0)).sum())} for s in ('train','validacion','prueba')}
        d[['Municipio','Parcela','Planta','Muestreo']].assign(conjunto=split).to_csv(out/'particion.csv',index=False)
        event('datos','diseno_'+design,summary)
        arms={};bundles={};validation_rows=[]
        for enhanced in (False,True):
            arm='mejorado' if enhanced else 'referencia';f=features(d,bank,enhanced)
            prep=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler())
            x=prep.fit_transform(f[tr]);xv=prep.transform(f[va])
            forest=RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=10,n_jobs=2,random_state=42).fit(x,y[tr]>0)
            p=forest.predict_proba(xv)[:,1];t98=recall_threshold(y[va],p)
            tgrid=np.arange(.05,.951,.01)
            tbalanced=max(tgrid,key=lambda t:statistics(y[va],p,np.zeros(va.sum()),t)['exactitud_balanceada'])
            positive_pp=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),PCA(n_components=.95,svd_solver='full'))
            a=positive_pp.fit_transform(f[positive]);zv=positive_pp.transform(f[va])
            log=quantity_model('log1p_lineal').fit(a,np.log1p(y[positive]));logv=quantity_predict(log,'log1p_lineal',zv)
            seeds=(42,43,44) if enhanced else (43,);candidates=[]
            for seed in seeds:
                gp=SymbolicRegressor3(seed=seed).fit(a,y[positive]);gpv=gp.predict(zv)
                weights=np.linspace(0,1,21) if enhanced else [1.]
                for w in weights:
                    q=np.where(p>=t98,w*gpv+(1-w)*logv,0.)
                    candidates.append({'semilla':seed,'peso_GP':float(w),'validacion':statistics(y[va],p,q,t98),'gp':gp})
                event('modelacion','mezcla_'+design+'_'+arm,{'semilla':seed,'umbral_98':t98,'componentes':a.shape[1],
                     'mejor_MAE':min(c['validacion']['MAE'] for c in candidates)})
            best=min(candidates,key=lambda c:c['validacion']['MAE'])
            b={'forest':forest,'presence_pp':prep,'positive_pp':positive_pp,'gp':best['gp'],'log':log,
               'threshold':t98,'gp_weight':best['peso_GP'],'enhanced':enhanced,'climate_bank':bank if enhanced else None,
               'lookup_geografia':source['lookup_geografia'],'feature_columns':list(f),'schema':'mejora_temporal_v1'}
            arms[arm]={'features':list(f),'componentes_positivos':a.shape[1],
                       'umbral_balanceado':float(tbalanced),'umbral_sensibilidad_98':t98,
                       'semilla_GP':best['semilla'],'peso_GP':best['peso_GP'],'validacion':best['validacion'],
                       'busqueda_mezcla':[{k:v for k,v in c.items() if k!='gp'} for c in candidates]}
            bundles[arm]=b
            for c in candidates:validation_rows.append({'rama':arm,'semilla':c['semilla'],'peso_GP':c['peso_GP'],**c['validacion']})
            event('estadistica','validacion_'+design+'_'+arm,arms[arm]['validacion'])
        base=arms['referencia']['validacion'];candidate=arms['mejorado']['validacion']
        passed=bool(candidate['MAE']<base['MAE']-1e-8 and candidate['sensibilidad']>=.98-1e-12 and candidate['FP']<=base['FP'])
        validation_gate.append({'diseno':design,'mejora_validacion':passed,'MAE_referencia':base['MAE'],
                'MAE_mejorado':candidate['MAE'],'FP_referencia':base['FP'],'FP_mejorado':candidate['FP'],
                'sensibilidad_mejorado':candidate['sensibilidad']})
        reports[design]={'split':summary,'ramas':arms,'mejora_validacion':passed}
        pd.DataFrame(validation_rows).to_csv(out/'busqueda_validacion.csv',index=False)
        for arm,b in bundles.items():
            result=predict(b,d[te]);test=statistics(y[te],result.probabilidad_presencia.to_numpy(),result.incidencia_predicha.to_numpy(),b['threshold'])
            reports[design]['ramas'][arm]['prueba']=test
            result['incidencia_observada']=y[te]
            for c in ['Municipio','Parcela','Planta','Muestreo']:result[c]=d.loc[te,c].to_numpy()
            result.to_csv(out/('predicciones_'+arm+'.csv'),index=False);joblib.dump(b,out/('modelo_'+arm+'.joblib'),compress=3)
            event('auditoria','prueba_'+design+'_'+arm,test)
        (out/'resultados.json').write_text(json.dumps(reports[design],ensure_ascii=False,indent=2))
    coverage=reports['temporal_12_5_7']['split']['prueba']['positivos']>0
    gate={'reglas':'Mejor MAE y FP no mayor a referencia con umbral de sensibilidad >=98%, en ambas validaciones; además cobertura de positivos en la prueba temporal para contrastar ambas fases.',
          'validaciones':validation_gate,'cobertura_positiva_prueba_temporal':coverage,
          'avanzar_arbol_difuso':bool(all(r['mejora_validacion'] for r in validation_gate) and coverage),
          'prueba_no_usada_para_optimizar':'Sólo se audita presencia de ambas clases; ningún MAE de prueba abre el avance.'}
    if gate['avanzar_arbol_difuso']:
        # La condición se evalúa antes de crear o entrenar el siguiente candidato.
        event('orquestador','avance_difuso_habilitado',gate)
    else:event('orquestador','arbol_difuso_no_ejecutado',gate)
    report={'disenos':reports,'condicion_avance':gate,
            'alcance':'Evaluación secuencial a un muestreo: se supone disponible el resultado del muestreo anterior, incluso dentro de validación/prueba. No es predicción abierta de varios muestreos.',
            'riesgos_datos':['Muestreos 17–24 totalmente cero; falta confirmar ausencia observada vs dato sin captura.',
                'Prueba temporal no contiene positivos: sensibilidad no estimable.',
                'Simulación original usa constantes anuales: disponibilidad histórica no acreditada; no es validación prospectiva.',
                'Nuevas parcelas se evalúan con antecedentes de sus plantas conocidos al predecir; no es arranque sin historia.'],
            'cambios':'Se excluyen todas las ventanas futuras y circulares previas de ambos brazos; referencia 28 variables. Mejorado añade dos retardos y 18 rasgos climáticos previos: 48 antes de imputación/indicadores.',
            'normalizacion':'Imputación por mediana, indicadores, escalado y ACP sólo train; ACP sólo positivos train.',
            'accion_difusa':'No entrenado: condición no satisfecha.' if not gate['avanzar_arbol_difuso'] else 'Habilitado para comparación posterior.'}
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (OUT/'condicion_avance.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':run()
