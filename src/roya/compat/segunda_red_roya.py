"""Bosque -> red1 de comunidades -> red2 geográfica/ventana15 -> seis regresiones.

Los coeficientes supervisados de bosque/red1 que producen entradas train de
red2 excluyen el fold exterior. El bosque OOF previo se reutiliza; red1 se
reajusta con un OOF interior. Umbral, arquitectura y política de red1 congelados.
"""
import json,warnings,hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,confusion_matrix
from threadpoolctl import threadpool_limits
import corrector_cuatro_estados as c

ROOT=c.ROOT
OUT=ROOT/'evaluacion_segunda_red'
REGRESSIONS=['lineal_multiple','log1p_lineal','polinomio_2','polinomio_3','ciclico','red_6_9_18_36_36']
LAGS=[f'cond_{stage}_dia_m{lag:02d}' for stage in ['germinacion','apresorio'] for lag in range(15,0,-1)]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def previous15(d,bank):
    """30 flags diarios y dos banderas de existencia; ventana incompleta=NaN."""
    d=d.reset_index(drop=True);out=pd.DataFrame(np.nan,index=d.index,columns=LAGS+['alguna_cond_germinacion_15d','alguna_cond_apresorio_15d'])
    out['ventana15_incompleta']=0.
    for period,ix in d.groupby(c.g.periods(d),sort=True).groups.items():
        local=d.loc[ix]
        if 'Fecha_muestreo_exacta' in local and local.Fecha_muestreo_exacta.notna().any():
            raise ValueError('Este perfil requiere bloques quincenales; fechas exactas requieren reconstruir las entradas de todas las etapas.')
        date=pd.Timestamp(period[:7]+'-'+('01' if period[-1]=='1' else '16'))
        if date not in bank['dates']:raise ValueError('Fecha fuera del año simulado: no se repite automáticamente.')
        day=bank['dates'].get_loc(date);ids=c.g.site_ids(local).map(bank['xy'])
        if ids.isna().any():raise ValueError('Ubicación sin ventana simulada original.')
        if day<15:out.loc[ix,'ventana15_incompleta']=1.;continue
        for stage in ['germinacion','apresorio']:
            response=bank['arrays'][stage+'_respuesta_realizacion_1'][day-15:day,ids.to_numpy(int)].T
            if not np.isfinite(response).all():raise ValueError('Faltan valores dentro de la ventana; no se convierten a cero.')
            flags=(response>=.9).astype(float)
            out.loc[ix,[f'cond_{stage}_dia_m{lag:02d}' for lag in range(15,0,-1)]]=flags
            out.loc[ix,'alguna_cond_'+stage+'_15d']=flags.any(axis=1).astype(float)
    return out

def last_features(d,pforest,p1,states,first,bank):
    d=d.reset_index(drop=True)
    f=pd.DataFrame({f'prob_red1_estado_{k}':states[:,i] for i,k in enumerate(c.CODES)})
    f['puntuacion_presencia_red1']=p1;f['decision_red1']=np.asarray(first,float)
    f['probabilidad_bosque']=pforest;f['decision_bosque']=(pforest>=bank['forest_threshold']).astype(float)
    for name in ['Coordenada_X','Coordenada_Y','Altitud']:f[name]=d[name].to_numpy()
    return pd.concat([f,previous15(d,bank['sim'])],axis=1)

def upstream_predict(upstream,d,m):
    p=upstream['forest'].predict_proba(d[c.FOREST_FEATURES])[:,1]
    base=p>=upstream['forest_threshold'];policy=upstream['policy']
    p1,states=c.conditional_prob(upstream['correctors'][policy['red']],c.meta_features(m,p,upstream['forest_threshold']),base)
    first=c.apply_policy(base,p1,policy)
    return p,base,p1,states,first

def nested_train_inputs(upstream,d,m,split,event):
    tr=split=='train';data=d.loc[tr].reset_index(drop=True);members=m.loc[tr].reset_index(drop=True)
    stored=pd.read_csv(c.OUT/'objetivos_OOF.csv')
    assert np.array_equal(stored.fila_excel.to_numpy(),data.fila_excel.to_numpy())
    labels=data[c.g.TARGET].to_numpy()>0;sites=c.g.site_ids(data).to_numpy();folds=stored.fold.to_numpy(int)
    poof=stored.probabilidad_OOF.to_numpy();p1oof=np.zeros(len(data));states_oof=np.zeros((len(data),4));first_oof=np.zeros(len(data),bool)
    inner_assign=[];notices={};policy=upstream['policy'];threshold=upstream['forest_threshold']
    power=upstream['correctors'][policy['red']]['peso_clases_exponente']
    for outer in sorted(np.unique(folds)):
        a=np.flatnonzero(folds!=outer);held=np.flatnonzero(folds==outer)
        assert not set(sites[a])&set(sites[held])
        inner=StratifiedGroupKFold(3,shuffle=True,random_state=100+int(outer));p_inner=np.zeros(len(a));inner_id=np.full(len(a),-1)
        for k,(fit,evaluate) in enumerate(inner.split(data.loc[a,c.FOREST_FEATURES],labels[a],sites[a])):
            assert not set(sites[a[fit]])&set(sites[a[evaluate]])
            model=c.forest(1000+int(outer)*10+k).fit(data.loc[a[fit],c.FOREST_FEATURES],labels[a[fit]])
            p_inner[evaluate]=model.predict_proba(data.loc[a[evaluate],c.FOREST_FEATURES])[:,1];inner_id[evaluate]=k
        inner_assign.append(pd.DataFrame(dict(outer=outer,sitio=sites[a],inner=inner_id)).drop_duplicates())
        code=c.truth_codes(labels[a],p_inner>=threshold)
        assert set(np.unique(code))==set(c.CODES)
        first_model=c.fit_corrector(c.meta_features(members.iloc[a],p_inner,threshold),code,power)
        p1,states=c.conditional_prob(first_model,c.meta_features(members.iloc[held],poof[held],threshold),poof[held]>=threshold)
        p1oof[held]=p1;states_oof[held]=states;first_oof[held]=c.apply_policy(poof[held]>=threshold,p1,policy)
        notices[str(outer)]=first_model['advertencias']
        event('OOF_anidado_red1',dict(outer=int(outer),filas_train=len(a),filas_reservadas=len(held),avisos=notices[str(outer)]))
    c.g.csv(pd.concat(inner_assign,ignore_index=True),OUT/'particiones_interiores.csv')
    f=last_features(data,poof,p1oof,states_oof,first_oof,dict(forest_threshold=threshold,sim=upstream['quantity']['bank']))
    table=pd.concat([data[['fila_excel','Coordenada_X','Coordenada_Y']],pd.DataFrame(dict(fold=folds,presencia_real=labels,
        decision_red1_OOF=first_oof,objetivo_error=c.truth_codes(labels,first_oof))),f],axis=1)
    # Coordenadas sólo se guardan una vez en esta tabla.
    table=table.loc[:,~table.columns.duplicated()]
    c.g.csv(table,OUT/'entradas_red2_OOF.csv')
    return f,c.truth_codes(labels,first_oof),notices

def fit_second(f,y,power):
    pre=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler());x=pre.fit_transform(f)
    labels,counts=np.unique(y,return_counts=True)
    if set(labels)!=set(c.CODES):raise ValueError('Se necesitan ejemplos OOF de los cuatro estados para entrenar red2.')
    weights={k:(len(y)/(4*n))**power for k,n in zip(labels,counts)};w=np.array([weights[k] for k in y]);w/=w.mean()
    model=MLPClassifier(hidden_layer_sizes=(18,12,6),alpha=1,max_iter=1000,early_stopping=False,random_state=42,tol=1e-4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(x,y,sample_weight=w)
    return dict(pre=pre,model=model,peso_clases_exponente=power,capas=[18,12,6],advertencias=[str(v.message) for v in caught])

def fit_binary_second(f,code,boost):
    pre=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler());x=pre.fit_transform(f)
    y=np.isin(code,[-1,1]);wrong=np.isin(code,[-1,2]);w=1+boost*wrong.astype(float);w/=w.mean()
    model=MLPClassifier(hidden_layer_sizes=(18,12,6),alpha=1,max_iter=1000,early_stopping=False,random_state=42,tol=1e-4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(x,y,sample_weight=w)
    return dict(pre=pre,model=model,tipo_objetivo='presencia_binaria',peso_errores_extra=boost,capas=[18,12,6],advertencias=[str(v.message) for v in caught])

def second_probability(model,f,first):
    if model.get('tipo_objetivo')!='presencia_binaria':return c.conditional_prob(model,f,first)
    probability=model['model'].predict_proba(model['pre'].transform(f))[:,list(model['model'].classes_).index(True)]
    states=np.zeros((len(f),4));first=np.asarray(first,bool)
    states[~first,0]=probability[~first];states[~first,1]=1-probability[~first]
    states[first,2]=probability[first];states[first,3]=1-probability[first]
    return probability,states

def run(exchange=None):
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    def event(action,evidence):
        print(action,str(evidence)[:900],flush=True)
        with (OUT/'eventos.jsonl').open('a') as s:s.write(json.dumps(dict(accion=action,evidencia=evidence),ensure_ascii=False)+'\n')
    upstream=joblib.load(c.OUT/'modelo.joblib')
    d=c.g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(c.g.OUT/'comunidades_por_observacion.csv');assert np.array_equal(d.fila_excel,m.fila_excel)
    split=m.conjunto.to_numpy();masks={s:split==s for s in ['train','validacion','prueba']};y=d[c.g.TARGET].to_numpy(float);yb=y>0
    train_x,code,nested_notices=nested_train_inputs(upstream,d,m,split,event)
    event('objetivos_red2',dict(conteos={str(k):int((code==k).sum()) for k in c.CODES},entradas=len(train_x.columns)))
    pforest,forest,p1,states,first=upstream_predict(upstream,d,m)
    f=last_features(d,pforest,p1,states,first,dict(forest_threshold=upstream['forest_threshold'],sim=upstream['quantity']['bank']))
    va=masks['validacion'];te=masks['prueba'];baseline=c.g.stats(yb[va],first[va]);nets={};rows=[];val_scores={}
    for power in [0.,.5,1.]:
        name='red2_peso_'+format(power,'.2f');model=fit_second(train_x,code,power);nets[name]=model
        prob,_=second_probability(model,f[va],first[va]);val_scores[name]=prob
        if not model['advertencias']:rows+=c.policies(yb[va],first[va],prob,name)
        event('red2_entrenada',dict(nombre=name,avisos=model['advertencias']))
    for boost in [0.,2.,5.]:
        name='red2_binaria_error_'+format(boost,'.0f');model=fit_binary_second(train_x,code,boost);nets[name]=model
        prob,_=second_probability(model,f[va],first[va]);val_scores[name]=prob
        if not model['advertencias']:rows+=c.policies(yb[va],first[va],prob,name)
        event('red2_binaria_entrenada',dict(nombre=name,avisos=model['advertencias']))
    if not rows:raise RuntimeError('No hay segunda red convergente elegible.')
    selected=c.choose_policy(rows,baseline);experimental=max(rows,key=lambda v:(v['exactitud_balanceada'],-v['FP'],-v['FN']))
    operational=selected or dict(red=experimental['red'],umbral_negativos=1.01,umbral_positivos=0.,**baseline)
    selection=dict(adoptar=selected is not None,politica=operational,experimental=experimental,baseline=baseline,
        criterio='Red2 se adopta si validación no aumenta FP ni FN respecto a red1, mejora al menos uno y exactitud balanceada >=0.001. Si no, identidad como política conservadora; experimental disponible.')
    c.g.csv(pd.DataFrame(rows),OUT/'politicas_validacion.csv');(OUT/'seleccion_antes_prueba.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    if exchange:exchange.validation_ready(selection)
    event('seleccion_congelada',selection)
    comparison=[];effects=[];predictions=[];finals={};class_matrices={}
    for s in ['validacion','prueba']:
        mask=masks[s];prob,classes=second_probability(nets[operational['red']],f[mask],first[mask])
        eprob,_=second_probability(nets[experimental['red']],f[mask],first[mask]);final=c.apply_policy(first[mask],prob,operational);trial=c.apply_policy(first[mask],eprob,experimental);finals[s]=final
        for name,value in [('bosque',forest[mask]),('bosque_red1',first[mask]),('bosque_red1_red2',final),('red2_experimental',trial)]:comparison.append(dict(conjunto=s,etapa=name,**c.g.stats(yb[mask],value)))
        inferred=np.where(first[mask],np.where(final,1,2),np.where(final,-1,0));actual=c.truth_codes(yb[mask],first[mask])
        class_matrices[s]=confusion_matrix(actual,inferred,labels=c.CODES).tolist()
        effects.append(dict(conjunto=s,
            omisiones_recuperadas=int(((~first[mask])&final&yb[mask]).sum()),
            nuevas_falsas_alarmas=int(((~first[mask])&final&(~yb[mask])).sum()),
            falsas_alarmas_eliminadas=int((first[mask]&(~final)&(~yb[mask])).sum()),
            nuevas_omisiones=int((first[mask]&(~final)&yb[mask]).sum())))
        t=d.loc[mask,['fila_excel','Municipio','Parcela','Mes_registrado','Quincena','Coordenada_X','Coordenada_Y',c.g.TARGET]].copy()
        t['conjunto']=s;t['probabilidad_bosque']=pforest[mask];t['presencia_bosque']=forest[mask];t['presencia_red1']=first[mask]
        t['puntuacion_red1']=p1[mask];t['puntuacion_red2']=prob;t['presencia_predicha']=final;t['presencia_experimental']=trial
        t['codigo_error_red1_real']=actual;t['codigo_error_red1_estimado']=inferred
        for i,k in enumerate(c.CODES):t['prob_red2_estado_'+str(k)]=classes[:,i]
        predictions.append(t)
    # Seis familias previas: se conserva la estimación positiva; sólo cambia la
    # puerta de presencia. Programación genética permanece en perfiles históricos.
    quantity=upstream['quantity'];quantity['regressors']={k:v for k,v in quantity['regressors'].items() if k in REGRESSIONS}
    assert set(quantity['regressors'])==set(REGRESSIONS)
    z=quantity['reg_representation'].transform(c.g.tabular(d,m,'combinadas'));q_rows=[]
    for name,model in quantity['regressors'].items():
        for s in ['validacion','prueba']:
            mask=masks[s];q=c.g.reg_predict(model,name,z[mask],d[mask]);pred=np.where(finals[s],q,0)
            q_rows.append(dict(modelo=name,conjunto=s,MAE=float(mean_absolute_error(y[mask],pred)),RMSE=float(np.sqrt(mean_squared_error(y[mask],pred)))))
    chosen=min([v for v in q_rows if v['conjunto']=='validacion'],key=lambda v:v['MAE'])['modelo'];quantity['selected_regression']=chosen
    magnitude=[]
    for s,t in zip(['validacion','prueba'],predictions):
        mask=masks[s];q=c.g.reg_predict(quantity['regressors'][chosen],chosen,z[mask],d[mask]);t['incidencia_predicha']=np.where(finals[s],q,0)
        for name,gate in [('bosque',forest[mask]),('bosque_red1',first[mask]),('bosque_red1_red2',finals[s])]:
            value=np.where(gate,q,0);magnitude.append(dict(conjunto=s,etapa=name,modelo_cantidad=chosen,MAE=float(mean_absolute_error(y[mask],value)),RMSE=float(np.sqrt(mean_squared_error(y[mask],value)))))
    c.g.csv(pd.concat(predictions),OUT/'predicciones.csv');c.g.csv(pd.DataFrame(comparison),OUT/'comparacion.csv');c.g.csv(pd.DataFrame(effects),OUT/'efectos_red2.csv')
    c.g.csv(pd.DataFrame(q_rows),OUT/'seis_regresiones.csv');c.g.csv(pd.DataFrame(magnitude),OUT/'magnitud_por_etapa.csv')
    # Guardar sólo red1 seleccionada y seis regresiones, conservando el artefacto
    # histórico de cuatro estados intacto.
    upstream['correctors']={upstream['policy']['red']:upstream['correctors'][upstream['policy']['red']]}
    bundle=dict(upstream=upstream,second_models=nets,policy=operational,experimental_policy=experimental,
        selection=selection,feature_names=list(f),quantity=quantity)
    report=dict(cadena='bosque clima/cultivo -> red1 de comunidades -> red2 de salidas, geografía, altitud y15d -> seis regresiones',
        diseno='Misma partición50/20/30 por coordenadas. Entradas train de red2:5 folds exteriores y3 interiores para generar red1 OOF. Configuración de red1 congelada del experimento previo.',
        nested_supervision='Cada red1 exterior aprende de predicciones RF interiores sin las etiquetas del fold exterior. Las redes ambientales usan covariables de todo train sin incidencia. Arquitecturas/umbrales previamente elegidos no se reoptimizan por fold.',
        feature_names=list(f),arquitectura_red2=[18,12,6],formulaciones_red2=['cuatro_estados','presencia_binaria_con_peso_en_errores'],objetivos_train={str(k):int((code==k).sum()) for k in c.CODES},
        seleccion=selection,comparacion=comparison,efectos=effects,matrices_estados=class_matrices,regresiones=q_rows,
        regresion_elegida=chosen,magnitud=magnitude,advertencias_oof_red1=nested_notices,
        advertencias_red2={k:v['advertencias'] for k,v in nets.items()},
        procedencia=dict(modelo_red1_sha256=sha(c.OUT/'modelo.joblib'),oof_bosque_sha256=sha(c.OUT/'objetivos_OOF.csv'),comunidades_sha256=sha(c.g.OUT/'comunidades_por_observacion.csv')),
        limites=['Histórico exploratorio consultado varias veces; no prueba prospectiva independiente.',
            'Muestreos17–24 completamente cero: confirmar ausencia observada frente a falta de captura. Unidad de incidencia pendiente.',
            'Coordenadas X/Y proyectadas, CRS EPSG32614 provisional; altitud es la columna original Altitud. Diferencias con MDE pendientes.',
            'Los15d son aptitud térmica simulada para germinación/apresorio; umbral0.9 heredado, no infección comprobada ni observación de mojado foliar.',
            'Fecha de referencia: primer día posible de quincena. Hoy excluido; la primera quincena incompleta conserva ausentes y un indicador, nunca se rellena como ausencia de condiciones.',
            'Probabilidades neuronales son puntuaciones sin calibración independiente. Añadir una etapa puede introducir errores nuevos.',
            'Los seis modelos de magnitud reutilizan coeficientes previamente ajustados sólo con positivos train; se reevalúan y seleccionan por MAE total de validación. GP se excluye de esta ruta por la petición de seis familias.'])
    experimental_quantity(report,bundle)
    c.g.save_model(bundle,OUT/'modelo.joblib')
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));make_report(report);plot(report)
    event('finalizado',dict(adoptar=selection['adoptar'],regresion=chosen))
    return report

def predict(bundle,d,variant='recomendado'):
    upstream=bundle['upstream'];d=c.g.enrich(upstream['quantity'],d).reset_index(drop=True)
    m=c.g.graph_transform(d,upstream['quantity']['bank'],upstream['quantity']['blocks'])
    p,forest,p1,states,first=upstream_predict(upstream,d,m)
    f=last_features(d,p,p1,states,first,dict(forest_threshold=upstream['forest_threshold'],sim=upstream['quantity']['bank']))
    policy=bundle['experimental_policy'] if variant=='experimental' else bundle['policy']
    p2,_=second_probability(bundle['second_models'][policy['red']],f,first);final=c.apply_policy(first,p2,policy)
    if variant=='red1':final=first
    elif variant=='bosque':final=forest
    quantity=bundle['quantity'];name=quantity.get('selected_regression_experimental',quantity['selected_regression']) if variant=='experimental' else quantity['selected_regression'];z=quantity['reg_representation'].transform(c.g.tabular(d,m,'combinadas'))
    q=np.where(final,c.g.reg_predict(quantity['regressors'][name],name,z,d),0)
    return pd.DataFrame(dict(probabilidad_bosque=p,puntuacion_red1=p1,puntuacion_red2=p2,
        presencia_predicha=final,grupo_final=m.comunidad_fusion,incidencia_predicha=q,estado=np.where(final,'cantidad_estimada','sin_roya')))

def experimental_quantity(report,bundle):
    """Evalúa también las seis familias con red2 activa, sin alterar su política."""
    d=c.g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(c.g.OUT/'comunidades_por_observacion.csv');pred=pd.read_csv(OUT/'predicciones.csv').set_index('fila_excel')
    quantity=bundle['quantity'];z=quantity['reg_representation'].transform(c.g.tabular(d,m,'combinadas'));rows=[]
    for name,model in quantity['regressors'].items():
        for tag in ['validacion','prueba']:
            mask=m.conjunto.to_numpy()==tag;gate=pred.loc[d.loc[mask,'fila_excel'],'presencia_experimental'].to_numpy()
            q=np.where(gate,c.g.reg_predict(model,name,z[mask],d[mask]),0)
            rows.append(dict(modelo=name,conjunto=tag,MAE=float(mean_absolute_error(d.loc[mask,c.g.TARGET],q)),RMSE=float(np.sqrt(mean_squared_error(d.loc[mask,c.g.TARGET],q)))))
    chosen=min([v for v in rows if v['conjunto']=='validacion'],key=lambda v:v['MAE'])['modelo']
    quantity['selected_regression_experimental']=chosen;report['regresiones_experimental']=rows;report['regresion_elegida_experimental']=chosen
    c.g.csv(pd.DataFrame(rows),OUT/'seis_regresiones_experimental.csv')

def make_report(r):
    lines=['# Bosque, primera red y segunda red final','',r['cadena'],'','## Clasificación','',pd.DataFrame(r['comparacion']).to_markdown(index=False),'',
        'Segunda red adoptada en validación: **'+str(r['seleccion']['adoptar'])+'**. '+r['seleccion']['criterio'],'',
        '## Cambios introducidos por red2','',pd.DataFrame(r['efectos']).to_markdown(index=False),'',
        '## Entrenamiento y entradas','',r['diseno'],'',r['nested_supervision'],'',
        'Red1 queda congelada en su configuración6–9–12 y red2 usa18–12–6, regularización alpha1 y entradas estandarizadas sólo con train. Se comparan ponderaciones0,0.5,1 de las cuatro clases de error y tres variantes binarias con multiplicadores1,3,6 para ejemplos donde falla red1. Las clases se calculan respecto a red1:0=TN,−1=FN,1=TP,2=FP; son objetivos/auditoría, nunca entradas reales. La variante binaria predice presencia directamente y sus probabilidades de error se derivan usando la decisión conocida de red1.','',
        'Red2 recibe las cuatro probabilidades de red1, su puntuación y decisión; probabilidad y decisión del bosque; X,Y,Altitud;30 indicadores diarios, dos indicadores de alguna condición en15d y ventana incompleta. Estados incompatibles con red1 se enmascaran y renormalizan.','',
        'Entradas exactas: '+', '.join(r['feature_names']), '','## Seis modelos de cantidad','',pd.DataFrame(r['regresiones']).to_markdown(index=False),'',
        'Elegido por validación: **'+r['regresion_elegida']+'**. Para aislar el efecto de la segunda red, la tabla siguiente aplica la misma regresión a las tres puertas de presencia.','',pd.DataFrame(r['magnitud']).to_markdown(index=False),'',
        'Se conservan los perfiles históricos con programación genética; esta ruta incluye lineal, logarítmico, polinomios2/3, cíclico y neuronal.','',
        '## Orquestación','',
        '`orquestador_segunda_red.py` coordina datos, modelación, estadística, auditoría y memoria. El estadístico reproduce la selección con validación y el auditor impide usar prueba para cambiarla. La memoria registra el nuevo experimento junto a los anteriores; no repite búsquedas idénticas. Son roles computacionales, no agentes LLM.','',
        '## Límites','']+['- '+v for v in r['limites']]
    if r.get('regresiones_experimental'):
        lines+=['','## Seis regresiones con la segunda red activa','',pd.DataFrame(r['regresiones_experimental']).to_markdown(index=False),'',
            'Familia seleccionada por validación en la ruta experimental: '+r['regresion_elegida_experimental']+'. Esto permite examinar la cadena completa sin confundirla con la política recomendada que no adopta red2.']
    (OUT/'INFORME.md').write_text('\n'.join(lines))

def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[v for v in r['comparacion'] if v['conjunto']=='prueba'];fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    for ax,v in zip(axes,rows):
        a=np.array([[v['TN'],v['FP']],[v['FN'],v['TP']]]);ax.imshow(a,cmap='Blues');title={'bosque':'Bosque','bosque_red1':'Bosque y red1','bosque_red1_red2':'Política conservadora','red2_experimental':'Red2 activa (experimental)'}[v['etapa']];ax.set_title(title,fontsize=10);ax.set_xticks([0,1],['Sin roya','Con roya']);ax.set_yticks([0,1],['Sin roya','Con roya']);ax.set_xlabel('Predicción');ax.set_ylabel('Registro observado')
        for (i,j),n in np.ndenumerate(a):ax.text(j,i,str(n),ha='center',va='center',fontsize=13,color='white' if n>a.max()/2 else 'black')
    fig.suptitle('Comparación por etapa · misma prueba por coordenadas');fig.savefig(OUT/'matrices_confusion.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    from segunda_red_roya import run as canonical_run
    canonical_run()
