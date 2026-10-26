"""Bosque binario congelado y regresiones de positivos: redes 3/4/5 + GP."""
import json,warnings
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble,inputs,splitter,quantity_model,quantity_predict,TARGET
from bosque_bayes_regresion import ROOT,enrich,metrics,NAMES
from probar_arbitro_logistico import cycle
from programacion_genetica_roya import SymbolicRegressor3,levels,size

OUT=ROOT/'evaluacion_bosque_regresiones_gp'

def amount(model,name,z,d):
    a=np.column_stack([z,cycle(d)]) if name=='ciclico' else z
    return quantity_predict(model,name,a)

def predict(b,d,name=None):
    name=b['selected_model'] if name is None else name
    d=enrich(b,d);f=inputs(d,b['simulation_bank']);z=b['positive_pp'].transform(f)
    p=b['detector'].predict_proba(b['presence_pp'].transform(f))[:,1];positive=p>=b['threshold']
    q=np.zeros(len(d))
    if positive.any():
        idx=np.flatnonzero(positive);q[positive]=amount(b['models'][name],name,z[positive],d.iloc[idx])
    return pd.DataFrame({'probabilidad_presencia':p,'presencia_predicha':positive,
        'grupo_final':positive.astype(int),'incidencia_predicha':q,
        'modelo_cantidad':name,'estado':np.where(positive,'cantidad_global_sin_grupos','sin_roya')})

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    def event(role,action,evidence):
        row={'fecha':datetime.now(timezone.utc).isoformat(),'agente':role,'accion':action,'evidencia':evidence}
        with (OUT/'eventos.jsonl').open('a') as s:s.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(action,str(evidence)[:600],flush=True)
    source=joblib.load(ROOT/'evaluacion_bosque_bayes/modelo.joblib')
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=splitter(d);pd.testing.assert_frame_equal(assignment,source['particion_plantas'])
    f=inputs(d,source['simulation_bank']);y=d[TARGET].to_numpy()
    pos=(split=='train')&(y>0);vp=(split=='validacion')&(y>0);te=split=='prueba';va=split=='validacion'
    z=source['positive_pp'].transform(f)
    b={k:source[k] for k in ['detector','threshold','presence_pp','positive_pp','simulation_bank','lookup_geografia','particion_plantas']}
    b.update(models={},selected_model='',schema='bosque_regresiones_gp_v1')
    rows=[];gp_reports=[]
    names=NAMES+['genetica_3_niveles_s'+str(seed) for seed in (42,43,44)]
    for name in names:
        model=SymbolicRegressor3(seed=int(name.rsplit('s',1)[1])) if name.startswith('genetica_') else quantity_model(name)
        a=np.column_stack([z[pos],cycle(d[pos])]) if name=='ciclico' else z[pos]
        target=np.log1p(y[pos]) if name=='log1p_lineal' else y[pos]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(a,target)
        notices=[str(w.message) for w in caught]
        b['models'][name]=model
        conditional=amount(model,name,z[vp],d[vp])
        result=predict(b,d[va],name)
        row={'modelo':name,'elegible':not notices,'advertencias':notices,
             'validacion_completa':metrics(y[va],result,b['threshold']),
             'MAE_validacion_positivos':float(mean_absolute_error(y[vp],conditional)),
             'RMSE_validacion_positivos':float(np.sqrt(mean_squared_error(y[vp],conditional)))}
        if name.startswith('genetica_'):
            gp_reports.append({'modelo':name,'semilla':model.seed,'arbol':model.tree_,
                'expresion':model.expression_,'centro_y':model.center_,'escala_y':model.scale_,
                'niveles':levels(model.tree_),'nodos':size(model.tree_),'max_niveles_evaluados':model.max_levels_seen_,
                'programas_evaluados':model.programs_evaluated_,'historia':model.history_})
        rows.append(row);event('modelacion','regresion_positivos',row)
    eligible=[r for r in rows if r['elegible']]
    selected=min(eligible,key=lambda r:r['validacion_completa']['MAE_cantidad'])['modelo'];b['selected_model']=selected
    testing=[]
    for row in rows:
        name=row['modelo'];result=predict(b,d[te],name);tp=te&(y>0)
        conditional=amount(b['models'][name],name,z[tp],d[tp])
        testing.append({'modelo':name,'elegible':row['elegible'],
            **metrics(y[te],result,b['threshold']),
            'MAE_positivos_sin_errores_del_bosque':float(mean_absolute_error(y[tp],conditional)),
            'RMSE_positivos_sin_errores_del_bosque':float(np.sqrt(mean_squared_error(y[tp],conditional)))})
        result['incidencia_observada']=y[te]
        for c in ['Municipio','Parcela','Planta','Mes_registrado','Muestreo']:result[c]=d.loc[te,c].to_numpy()
        result.to_csv(OUT/('predicciones_'+name+'.csv'),index=False)
    gp_candidates=[r for r in rows if r['modelo'].startswith('genetica_')]
    selected_gp=min(gp_candidates,key=lambda r:r['validacion_completa']['MAE_cantidad'])['modelo'];b['selected_gp']=selected_gp
    report={'detector_elegido':'bosque_reentrenado_congelado','umbral':b['threshold'],
            'diseno':'50/20/30 por plantas; mismos conjuntos previos. Regresiones sólo con positivos train.',
            'componentes_presencia':f.shape[1],'componentes_positivos':z.shape[1],
            'features':list(f),'seleccion_validacion':selected,'seleccion_gp_validacion':selected_gp,
            'comparacion_validacion':rows,'comparacion_prueba':testing,'programacion_genetica':gp_reports,
            'prueba':next(r for r in testing if r['modelo']==selected),
            'criterio':'MAE del sistema completo en validación, sin advertencias; prueba sólo informativa.',
            'arquitecturas':{'tres_capas':[6,9,12],'cuatro_capas':[6,9,18,36],'cinco_capas':[6,9,18,36,36],
                            'referencia_tres_capas':[10,10,10]},
            'gp_config':{'poblacion':256,'generaciones':40,'semillas':[42,43,44],
               'max_niveles':3,'raiz_es_nivel':1,'torneo':5,'elitismo':8,
               'prob_cruce':.70,'prob_mutacion':.25,'prob_reproduccion':.05,
               'penalizacion_por_nodo':.001,'objetivo':'MAE train + penalización por tamaño',
               'operadores':['+','-','*','division_protegida','sin','cos','log1p_abs'],
               'terminales':'16 CP estandarizados o constantes [-3,3]',
               'salida':'max(1,mediana_train_y + std_train_y * programa(Z)); Z escalado sólo con train positivos.'},
            'limites':['Prueba ya consultada: exploratoria; no nuevas parcelas ni validación temporal.',
                       'GP implementado localmente: límite de profundidad y evolución verificados con casos conocidos.',
                       'El bosque y ACP se reutilizan congelados para aislar la comparación de regresores.',
                       'Errores condicionales con presencia conocida no equivalen al rendimiento operativo completo.',
                       'Días siguientes son escenario anual; fechas exactas de muestreo, CRS y altitud pendientes.']}
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    pd.DataFrame(testing).to_csv(OUT/'comparacion_prueba.csv',index=False)
    pd.DataFrame([{'modelo':r['modelo'],'elegible':r['elegible'],
                  'MAE_completo':r['validacion_completa']['MAE_cantidad'],
                  'MAE_positivos':r['MAE_validacion_positivos']} for r in rows]).to_csv(OUT/'comparacion_validacion.csv',index=False)
    joblib.dump(b,OUT/'modelo.joblib',compress=3)
    event('auditoria','seleccion_final',{'elegido':selected,'gp_elegido':selected_gp,'prueba':report['prueba']})
    return report

if __name__=='__main__':run()
