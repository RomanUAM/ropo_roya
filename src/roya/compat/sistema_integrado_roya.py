"""Unifica variantes sin reemplazar evidencia ni ocultar casos no validados."""
import argparse,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import confusion_matrix
from roya_dos_etapas import predict as neural_predict,features,amount,TARGET
from roya_logistica_probabilidad import predict as logistic_predict
from probar_consenso_red_bosque import predict as consensus_predict
from probar_arbitro_logistico import predict as arbiter_predict

ROOT=Path(__file__).resolve().parent
TEMPORAL_PROFILES={
 'mejora_temporal_experimental':('evaluacion_mejora_temporal/temporal_12_5_7/modelo_mejorado.joblib','Experimental: antecedentes y mezcla, entrenamiento temporal'),
 'mejora_parcelas_experimental':('evaluacion_mejora_temporal/parcelas_6_2_4/modelo_mejorado.joblib','Experimental: antecedentes y mezcla, parcelas reservadas')}

EXTRA_PROFILES={
 'bosque_regresiones_gp':('evaluacion_bosque_regresiones_gp/modelo.joblib','Bosque sí/no → mejor regresión validada, incluidas redes y genética'),
 'bosque_genetica3':('evaluacion_bosque_regresiones_gp/modelo.joblib','Bosque sí/no → programación genética de tres niveles'),
 'bosque_regresion_red3':('evaluacion_bosque_regresiones_gp/modelo.joblib','Bosque sí/no → regresión neuronal 6–9–12'),
 'bosque_regresion_red4':('evaluacion_bosque_regresiones_gp/modelo.joblib','Bosque sí/no → regresión neuronal 6–9–18–36'),
 'bosque_regresion_red5':('evaluacion_bosque_regresiones_gp/modelo.joblib','Bosque sí/no → regresión neuronal 6–9–18–36–36'),
 'bosque_bayes':('evaluacion_bosque_bayes/modelo.joblib','Bosque reentrenado: estrategia elegida en validación'),
 'bosque_bayes_grupos':('evaluacion_bosque_bayes/modelo.joblib','Bosque → grupos bayesianos sobre ACP → regresiones'),
 'bosque_bayes_neuronal':('evaluacion_bosque_bayes/modelo.joblib','Bosque → red para grupos sobre ACP → regresiones'),
 'red5_completas':('evaluacion_red5_completas/modelo.joblib','Red 6–9–18–36–36 con conteos y variables completas'),
 'red5_acp':('evaluacion_red5_acp/modelo.joblib','Red 6–9–18–36–36 con conteos y ACP'),
 'red4_completas':('evaluacion_red4_completas/modelo.joblib','Red 6–9–18–36 con conteos y variables completas'),
 'red4_acp':('evaluacion_red4_acp/modelo.joblib','Red 6–9–18–36 con conteos y ACP'),
 'modelo_50_20_30':('modelos/modelo_50_20_30.joblib','Referencia previa: ACP sin conteos'),
 'base_completas':('evaluacion_base_completas/modelo.joblib','Variables completas sin conteos: referencia controlada'),
 'conteos_acp':('evaluacion_conteos_acp/modelo.joblib','Conteos simulados + ACP: dos etapas y seis grupos'),
 'conteos_completas':('evaluacion_conteos_completas/modelo.joblib','Conteos simulados + variables completas: dos etapas y seis grupos')}

PROFILES={
 'red_binaria':('modelos/red_binaria.joblib','red binaria 5–5–5 + grupo + cantidad'),
 'bosque_multiclase':('modelos/consenso.joblib','bosque: cero y cinco grupos'),
 'consenso':('modelos/consenso.joblib','red y bosque positivos'),
 'arbitro_logistico':('modelos/arbitro_logistico.joblib','logistica en desacuerdos'),
 'arbitro_3_9_27':('modelos/arbitro_3_9_27.joblib','red 3–9–27 en desacuerdos'),
 'logistica_probabilidad':('modelos/logistica_probabilidad.joblib','logistica y cantidad con probabilidad'),
 'doble_red_svm':('modelos/doble_red_svm.joblib','dos redes; SVM decide en discrepancias de grupo'),
 'doble_red_arbitro_profundo':('modelos/arbitro_profundo.joblib','dos redes; arbitro 20–15–12–9–9–12'),
 'geografia_exploratoria':('modelos/geografia.joblib','red binaria + bosque con distancias geograficas provisionales')}

LOGISTIC_PROFILES={
 'filtro_logistico':('evaluacion_filtro_logistico/modelo.joblib','Bosque + probabilidad condicional: política elegida en validación'),
 'filtro_logistico_conservador':('evaluacion_filtro_logistico/modelo.joblib','Bosque + filtro logístico conservador experimental'),
 'filtro_logistico_balanceado':('evaluacion_filtro_logistico/modelo.joblib','Bosque + filtro logístico balanceado experimental')}

V15_OPTIONS={
 'ventana15':{},
 'ventana15_sin_confirmar':{'route':'sin_confirmacion'},
 'ventana15_logistica':{'route':'logistica_balanceada'},
 'ventana15_svm':{'route':'svm_lineal_balanceada'},
 'ventana15_logistica_conservadora':{'route':'logistica_conservadora'},
 'ventana15_svm_conservador':{'route':'svm_lineal_conservadora'}}
for _name in ['bosque','logistica','svm_lineal','red_6_9_12','red_6_9_18_36','red_6_9_18_36_36']:
    V15_OPTIONS['v15_pres_'+_name]={'primary':_name,'route':'sin_confirmacion'}
for _name in ['lineal','log1p_lineal','polinomio_2','polinomio_3','ciclico','red_10_10_10','red_6_9_12','red_6_9_18_36','red_6_9_18_36_36','mediana_grupo','genetica_3_niveles_s42','genetica_3_niveles_s43','genetica_3_niveles_s44']:
    V15_OPTIONS['v15_reg_'+_name]={'regression':_name}
V15_PROFILES={name:('evaluacion_15dias/modelo.joblib','15 días: '+name) for name in V15_OPTIONS}

V15_OPTIONS.update({'confirmador_red6':{},'confirmador_red6_conservador':{'route':'red6_conservadora'},'confirmador_red6_sin_filtro':{'route':'sin_confirmacion'},'confirmador_red6_balanceado':{'route':'red6_balanceada'}})
V15_PROFILES.update({name:('evaluacion_confirmador_red6/modelo.joblib','Confirmador 10–5–10–5–10–5: '+name) for name in ['confirmador_red6','confirmador_red6_conservador','confirmador_red6_sin_filtro','confirmador_red6_balanceado']})

BINARY_BAYES_PROFILES={
 'bayes_binario':('evaluacion_confirmador_bayes_binario/modelo.joblib','Bosque → confirmación bayesiana 0/1 → regresión'),
 'bayes_binario_balanceado':('evaluacion_confirmador_bayes_binario/modelo.joblib','Confirmador bayesiano binario: política balanceada')}

GP_CONFIRM_PROFILES={
 'confirmador_genetico3':('evaluacion_confirmador_genetico3/modelo.joblib','Confirmador genético binario de tres niveles'),
 'confirmador_genetico3_balanceado':('evaluacion_confirmador_genetico3/modelo.joblib','Confirmador genético: política balanceada')}

HUMIDITY_BNN_PROFILES={
    'tres_bosques_HR_red_bayesiana':('evaluacion_humedad_bnn/modelo.joblib','Tres bosques con HR75/80 → red bayesiana → dos regresiones (experimental)'),
    'tres_logisticas_HR_red_bayesiana':('evaluacion_logisticas_bnn/modelo.joblib','Tres logísticas con HR75/80 → red bayesiana → dos regresiones (experimental)'),
    'tres_GP_HR_red_bayesiana':('evaluacion_genetica_bnn/modelo.joblib','Tres modelos de programación genética → red bayesiana → dos regresiones (experimental)'),
}

FOUR_PROFILES={
    'cuatro_RF':('evaluacion_cuatro_arquitecturas/resultados.json','Cuatro rutas: RF + BNN + siete modelos de cantidad'),
    'cuatro_LR':('evaluacion_cuatro_arquitecturas/resultados.json','Cuatro rutas: logística binomial + BNN + siete modelos de cantidad'),
    'cuatro_GP':('evaluacion_cuatro_arquitecturas/resultados.json','Cuatro rutas: genética + BNN + siete modelos de cantidad'),
    'cuatro_MIXTA':('evaluacion_cuatro_arquitecturas/clasificadores_MIXTA.joblib','Cuatro rutas: mejor probabilidad RF/LR/GP + BNN + siete cantidades'),
    'cuatro_mejor_clasificacion':('evaluacion_cuatro_arquitecturas/resultados.json','Cuatro rutas: presencia elegida en validación'),
    'cuatro_mejor_nivel':('evaluacion_cuatro_arquitecturas/resultados.json','Cuatro rutas: nivel elegido en validación'),
}

INTEGRAL_PROFILES={
    'integral_RF':('evaluacion_integral_binaria_topografia/clasificadores_RF.joblib','Actual: bosques + coordinador + seis regresiones'),
    'integral_LR':('evaluacion_integral_binaria_topografia/clasificadores_LR.joblib','Actual: logísticas + coordinador + seis regresiones'),
    'integral_GP':('evaluacion_integral_binaria_topografia/clasificadores_GP.joblib','Actual: programación genética + coordinador + seis regresiones'),
    'integral_mejor_clasificacion':('evaluacion_integral_binaria_topografia/registro_modelo.json','Actual: arquitectura elegida por clasificación en validación'),
    'integral_mejor_nivel':('evaluacion_integral_binaria_topografia/registro_modelo.json','Actual: combinación elegida por error de nivel en validación'),
}

THREE_FOREST_PROFILES={
    'tres_bosques_bayes24':('evaluacion_tres_bosques24/modelo.joblib','Tres bosques → Bayes conjunto → dos regresiones (experimental; sin humedad simulada)'),
}

TWO_FOREST_PROFILES={
    'dos_bosques_bayes_balanceado':('evaluacion_dos_bosques_bayes/modelo.joblib','Dos bosques → consenso bayesiano activo (balanceado)'),
    'dos_bosques_bayes_conservador':('evaluacion_dos_bosques_bayes/modelo.joblib','Dos bosques: política conservadora y probabilidad bayesiana'),
    'dos_bosques_global':('evaluacion_dos_bosques_bayes/modelo.joblib','Control: bosque de tres comunidades'),
    'dos_bosques_individual':('evaluacion_dos_bosques_bayes/modelo.joblib','Control: bosque de clima y fenología'),
}

SECOND_PROFILES={
    'segunda_red_final':('evaluacion_segunda_red/modelo.joblib','Bosque → primera red → segunda red final → seis regresiones'),
    'segunda_red_control_red1':('evaluacion_segunda_red/modelo.joblib','Control: bosque y primera red, mismas seis regresiones'),
    'segunda_red_control_bosque':('evaluacion_segunda_red/modelo.joblib','Control: bosque solo, mismas seis regresiones'),
    'segunda_red_experimental':('evaluacion_segunda_red/modelo.joblib','Segunda red: política experimental de validación'),
}

ERROR4_PROFILES={
    'corrector_cuatro_estados':('evaluacion_corrector_cuatro_estados/modelo.joblib','Bosque clima/cultivo → corrección neuronal de cuatro estados'),
    'corrector_cuatro_estados_bosque':('evaluacion_corrector_cuatro_estados/modelo.joblib','Control: bosque sólo clima y cultivo'),
    'corrector_cuatro_estados_experimental':('evaluacion_corrector_cuatro_estados/modelo.joblib','Corrector de cuatro estados: política experimental'),
}

GRANULAR_PROFILES={
    'redes_granulares':('evaluacion_redes_granulares/modelo.joblib','Redes Mahalanobis–Louvain y cascada granular'),
    'redes_granulares_bosque':('evaluacion_redes_granulares/modelo.joblib','Redes Mahalanobis–Louvain: control con bosque solo'),
    'redes_granulares_filtro05':('evaluacion_redes_granulares/modelo.joblib','Diagnóstico: cascada con filtro fenológico 0.5 (no recomendado)'),
}

def load_data(path):
    path=Path(path)
    if path.suffix.lower()=='.csv':return pd.read_csv(path)
    raw=json.loads(path.read_text());return pd.DataFrame(raw['registros'] if isinstance(raw,dict) else raw)

def predict_profile(profile,df,root=ROOT,regression=None):
    if profile in FOUR_PROFILES:
        from cuatro_arquitecturas_roya import predict_registered
        return predict_registered(df,profile.replace('cuatro_',''),regression=regression)
    if profile in INTEGRAL_PROFILES:
        from reentrenamiento_integral_roya import predict_registered
        return predict_registered(df,profile.replace('integral_',''))
    if profile=='tres_GP_HR_red_bayesiana':
        from comparar_genetica_bnn import predict
        return predict(joblib.load(Path(root)/HUMIDITY_BNN_PROFILES[profile][0]),df)
    if profile in HUMIDITY_BNN_PROFILES:
        from roya_humedad_bnn import predict
        return predict(joblib.load(Path(root)/HUMIDITY_BNN_PROFILES[profile][0]),df)
    if profile in THREE_FOREST_PROFILES:
        from tres_bosques_bayes24 import predict
        return predict(joblib.load(Path(root)/THREE_FOREST_PROFILES[profile][0]),df)
    if profile in TWO_FOREST_PROFILES:
        from dos_bosques_consenso_bayes import predict
        bundle=joblib.load(Path(root)/TWO_FOREST_PROFILES[profile][0])
        variant={'dos_bosques_bayes_balanceado':'balanceado','dos_bosques_bayes_conservador':'recomendado','dos_bosques_global':'global','dos_bosques_individual':'individual'}[profile]
        return predict(bundle,df,variant=variant)
    if profile in SECOND_PROFILES:
        from segunda_red_roya import predict
        bundle=joblib.load(Path(root)/SECOND_PROFILES[profile][0])
        variant={'segunda_red_final':'recomendado','segunda_red_control_red1':'red1','segunda_red_control_bosque':'bosque','segunda_red_experimental':'experimental'}[profile]
        return predict(bundle,df,variant=variant)
    if profile in ERROR4_PROFILES:
        from corrector_cuatro_estados import predict
        bundle=joblib.load(Path(root)/ERROR4_PROFILES[profile][0])
        return predict(bundle,df,variant='bosque' if profile.endswith('_bosque') else ('experimental' if profile.endswith('_experimental') else 'recomendado'))
    if profile in GRANULAR_PROFILES:
        from redes_granulares_roya import predict
        bundle=joblib.load(Path(root)/GRANULAR_PROFILES[profile][0])
        return predict(bundle,df,stage='bosque' if profile.endswith('_bosque') else ('filtro05' if profile.endswith('_filtro05') else 'completa'))
    if profile in GP_CONFIRM_PROFILES:
        from confirmador_genetico3 import infer
        bundle=joblib.load(Path(root)/GP_CONFIRM_PROFILES[profile][0])
        return infer(bundle,df,route='genetica3_balanceada' if profile.endswith('balanceado') else 'genetica3_conservadora')
    if profile in BINARY_BAYES_PROFILES:
        from confirmador_bayes_binario import infer
        bundle=joblib.load(Path(root)/BINARY_BAYES_PROFILES[profile][0])
        return infer(bundle,df,route='bayes_balanceada' if profile.endswith('balanceado') else 'bayes_conservadora')
    if profile in V15_PROFILES:
        from experimento_15dias import predict
        return predict(joblib.load(Path(root)/V15_PROFILES[profile][0]),df,**V15_OPTIONS[profile])
    if profile in LOGISTIC_PROFILES:
        from filtro_logistico_condicional import predict
        bundle=joblib.load(Path(root)/LOGISTIC_PROFILES[profile][0])
        variant={'filtro_logistico':bundle['selected_variant'],'filtro_logistico_conservador':'conservadora','filtro_logistico_balanceado':'balanceada'}[profile]
        return predict(bundle,df,variant)
    if profile in TEMPORAL_PROFILES:
        from mejora_temporal_roya import predict
        return predict(joblib.load(Path(root)/TEMPORAL_PROFILES[profile][0]),df)
    if profile in ['bosque_regresiones_gp','bosque_genetica3','bosque_regresion_red3','bosque_regresion_red4','bosque_regresion_red5']:
        from bosque_regresiones_gp import predict
        bundle=joblib.load(Path(root)/EXTRA_PROFILES[profile][0])
        name={'bosque_regresiones_gp':bundle['selected_model'],'bosque_genetica3':bundle['selected_gp'],
              'bosque_regresion_red3':'red_6_9_12','bosque_regresion_red4':'red_6_9_18_36','bosque_regresion_red5':'red_6_9_18_36_36'}[profile]
        return predict(bundle,df,name)
    if profile.startswith('bosque_bayes'):
        from bosque_bayes_regresion import predict
        bundle=joblib.load(Path(root)/EXTRA_PROFILES[profile][0])
        variant={'bosque_bayes':bundle['selected_variant'],'bosque_bayes_grupos':bundle['selected_group_variant'],'bosque_bayes_neuronal':bundle['selected_neural_variant']}[profile]
        if variant is None:raise ValueError('No hay red neuronal convergida disponible para grupos.')
        return predict(bundle,df,variant)
    if profile in EXTRA_PROFILES:
        from modelo_roya_50_20_30 import predict
        return predict(joblib.load(Path(root)/EXTRA_PROFILES[profile][0]),df)
    path,_=PROFILES[profile];bundle=joblib.load(Path(root)/path)
    if profile=='geografia_exploratoria':
        from probar_variables_geograficas import predict
        return predict(bundle,df)
    if profile in ['doble_red_svm','doble_red_arbitro_profundo']:
        from doble_red_svm import predict
        return predict(bundle,df)
    if profile=='red_binaria':return neural_predict(bundle,df)
    if profile=='consenso':return consensus_predict(bundle,df).rename(columns={'presencia_consenso':'presencia_predicha'})
    if profile.startswith('arbitro_'):return arbiter_predict(bundle,df)
    if profile=='logistica_probabilidad':return logistic_predict(bundle,df,'con_probabilidad')
    x=bundle['presence_preprocess'].transform(features(df));cls=bundle['forest'].predict(x)
    base=bundle['cantidad_bundle'];z=base['positive_preprocess'].transform(features(df))
    q=np.zeros(len(df));state=np.full(len(df),'sin_roya',dtype=object)
    for j in range(base['clusters'].n_clusters):
        mask=cls==j+1
        if not mask.any():continue
        name=base['selected'].get(j)
        if name is None:q[mask]=np.nan;state[mask]='grupo_sin_modelo_validado'
        else:q[mask]=amount(base['models'][j][name],name,z[mask]);state[mask]='cantidad_estimada'
    return pd.DataFrame({'presencia_predicha':cls>0,'grupo_final':cls,'incidencia_predicha':q,'estado':state})

def evaluate_all(source,output,root=ROOT):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    df=load_data(source)
    # Mantiene intactos los muestreos usados anteriormente como prueba.
    order=df.Muestreo.str.extract(r'(\d+)')[0].astype(int)
    test=df[order>=15];truth=test[TARGET].to_numpy()>0;rows=[]
    def event(agent,action,evidence):
        with (out/'eventos.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(fecha=datetime.now(timezone.utc).isoformat(),
                agente=agent,accion=action,evidencia=evidence),ensure_ascii=False)+'\n')
    event('datos','identificar',{'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'n_prueba':len(test),'advertencia':'prueba reutilizada, exploratoria'})
    for name in PROFILES:
        event('orquestador','ejecutar_variante',name)
        pred=predict_profile(name,test,root)
        assert (pred.loc[~pred.presencia_predicha,'incidencia_predicha']==0).all()
        tn,fp,fn,tp=confusion_matrix(truth,pred.presencia_predicha,labels=[0,1]).ravel()
        row=dict(variante=name,TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
            sensibilidad=float(tp/(tp+fn)),falsas_alarmas=float(fp/(fp+tn)),
            sin_cantidad_validada=int(pred.incidencia_predicha.isna().sum()))
        pred['incidencia_observada']=test[TARGET].to_numpy()
        pred.to_csv(out/(name+'.csv'),index=False);rows.append(row)
        event('estadistica','comparar',row)
    pd.DataFrame(rows).to_csv(out/'comparacion.csv',index=False)
    report={'estado':'exploratorio_no_validado','comparacion':rows,
        'regla':'No cambiar automaticamente la variante por resultados de prueba.',
        'pendientes':['Verificar ceros y unidad de incidencia','Obtener positivos de validacion por grupo',
                      'Evaluar otro ciclo anual y parcelas independientes']}
    if 'geografia_exploratoria' in PROFILES:
        report['pendientes'].extend(['Confirmar CRS/datum de parcelas antes de interpretar distancias',
            'Completar capas de fuentes de agua, cerros, construcciones y centros urbanos'])
        event('datos','cobertura_geografica_provisional',{'crs':'EPSG:32614 provisional',
            'distancias_usadas':['cauce cartografiado','camino cartografiado','zona residencial cartografiada'],
            'cerros_y_centros_urbanos_verificados':'pendientes; no rellenados con cero'})
    terrain_path=Path(root)/'evidencia/auditoria_barreras.json'
    if terrain_path.exists():
        terrain=json.loads(terrain_path.read_text())
        event('datos','auditar_relieve_y_estructuras',terrain)
        report['auditoria_barreras']=terrain
        report['pendientes'].extend(['Resolver diferencias entre altitud de Excel y MDE antes de interpretar relieve',
            'Validar barreras como modificadores de conectividad; no tratar fallas geologicas como muros'])
    (out/'dictamen.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    event('auditoria','entregar_dictamen',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('datos');p.add_argument('salida')
    a=p.parse_args();print(json.dumps(evaluate_all(a.datos,a.salida),ensure_ascii=False))
