"""Bosque clima/cultivo -> corrector neuronal de TN/FN/TP/FP con comunidades.

La clase de error sólo se conoce al entrenar/evaluar; jamás se usa de entrada.
Se generan objetivos con predicciones OOF por coordenadas, no con ajuste train.
"""
import json,warnings,hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
import redes_granulares_roya as g

ROOT=g.ROOT
OUT=ROOT/'evaluacion_corrector_cuatro_estados'
CLIMATE=['Temperatura promedio','Punto_de_rocio','Temperatura','Humedad_especifica','Humedad_relativa','Precipitacion']
FOREST_FEATURES=CLIMATE+g.STAGES
CODES=[-1,0,1,2]
MEANING={'-1':'falso_negativo','0':'negativo_correcto','1':'positivo_correcto','2':'falso_positivo'}

def truth_codes(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool)
    return np.where(p,np.where(y,1,2),np.where(y,-1,0))

def forest(seed=42):
    return make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(
        n_estimators=180,max_depth=12,min_samples_leaf=15,class_weight='balanced_subsample',n_jobs=2,random_state=seed))

def meta_features(m,p,threshold):
    f=m[[c for c in m if c.startswith('comunidad_')]+['ventana15_incompleta']].reset_index(drop=True).copy()
    f['probabilidad_bosque']=p;f['decision_bosque']=(p>=threshold).astype(float)
    return f

def fit_corrector(f,code,power):
    pre=g.preprocessor(f);x=pre.fit_transform(f)
    labels,counts=np.unique(code,return_counts=True);weights={k:(len(code)/(len(labels)*n))**power for k,n in zip(labels,counts)}
    w=np.array([weights[k] for k in code]);w/=w.mean()
    model=MLPClassifier(hidden_layer_sizes=(6,9,12),alpha=1,max_iter=1000,
        early_stopping=False,random_state=42,tol=1e-4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(x,code,sample_weight=w)
    return dict(pre=pre,model=model,peso_clases_exponente=power,
        advertencias=[str(v.message) for v in caught],capas=[6,9,12])

def conditional_prob(model,f,base):
    """Máscara de estados incompatibles con la decisión conocida del bosque."""
    raw=model['model'].predict_proba(model['pre'].transform(f))
    p=np.column_stack([raw[:,list(model['model'].classes_).index(k)] for k in CODES])
    # Bosque negativo sólo puede ser TN/FN; positivo sólo TP/FP.
    p[np.asarray(base),:2]=0;p[~np.asarray(base),2:]=0
    denominator=p.sum(axis=1,keepdims=True)
    if (denominator<=0).any():raise ValueError('Probabilidad nula en todos los estados compatibles.')
    p/=denominator
    presence=np.where(base,p[:,2],p[:,0])
    return presence,p

def apply_policy(base,p,policy):
    return np.where(base,p>=policy['umbral_positivos'],p>=policy['umbral_negativos'])

def policies(y,base,p,name):
    rows=[]
    # Incluye explícitamente conservar cada rama del bosque.
    negatives=np.r_[np.arange(.05,.951,.025),1.01]
    positives=np.r_[0.,np.arange(.05,.951,.025)]
    for tn in negatives:
        for tp in positives:
            prediction=apply_policy(base,p,dict(umbral_negativos=tn,umbral_positivos=tp))
            rows.append(dict(red=name,umbral_negativos=float(tn),umbral_positivos=float(tp),**g.stats(y,prediction)))
    return rows

def choose_policy(rows,baseline):
    good=[v for v in rows if v['FP']<=baseline['FP'] and v['FN']<=baseline['FN']
        and (v['FP']<baseline['FP'] or v['FN']<baseline['FN'])
        and v['exactitud_balanceada']>=baseline['exactitud_balanceada']+.001]
    if not good:return None
    return max(good,key=lambda v:(v['exactitud_balanceada'],-v['FP'],-v['FN']))

def run(exchange=None,plan=None):
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    def event(action,evidence):
        print(action,str(evidence)[:800],flush=True)
        with (OUT/'eventos.jsonl').open('a') as s:s.write(json.dumps(dict(accion=action,evidencia=evidence),ensure_ascii=False)+'\n')
    d=g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(g.OUT/'comunidades_por_observacion.csv')
    assert np.array_equal(d.fila_excel.to_numpy(),m.fila_excel.to_numpy())
    split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba'
    y=d[g.TARGET].to_numpy(float);yb=y>0;f=d[FOREST_FEATURES]
    # Umbral del bosque determinado sólo con predicciones OOF de train.
    folds=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    oof=np.zeros(tr.sum());fold_id=np.full(tr.sum(),-1);groups=g.site_ids(d[tr]).to_numpy();train_y=yb[tr]
    for k,(a,b) in enumerate(folds.split(f[tr],train_y,groups)):
        assert not set(groups[a])&set(groups[b])
        model=forest(42+k).fit(f[tr].iloc[a],train_y[a]);oof[b]=model.predict_proba(f[tr].iloc[b])[:,1];fold_id[b]=k
        event('bosque_OOF',dict(fold=k,train=len(a),evaluados=len(b)))
    candidates=np.arange(.1,.901,.025)
    threshold=float(max(candidates,key=lambda t:balanced_accuracy_score(train_y,oof>=t)))
    code=truth_codes(train_y,oof>=threshold)
    assert set(np.unique(code))==set(CODES)
    event('objetivos_cuatro_estados',dict(umbral_train=threshold,conteos={str(k):int((code==k).sum()) for k in CODES}))
    full_forest=forest().fit(f[tr],yb[tr]);pforest=full_forest.predict_proba(f)[:,1];base=pforest>=threshold
    meta_oof=meta_features(m[tr],oof,threshold);meta_all=meta_features(m,pforest,threshold)
    val_baseline=g.stats(yb[va],base[va]);nets={};all_policies=[];scores={}
    exponents=(plan or {}).get('exponentes',[0.,.5,1.])
    for power in exponents:
        name='red_peso_'+format(power,'.2f');net=fit_corrector(meta_oof,code,power);nets[name]=net
        val_p,val_classes=conditional_prob(net,meta_all[va],base[va]);scores[name]=dict(val_p=val_p,val_classes=val_classes)
        if not net['advertencias']:all_policies+=policies(yb[va],base[va],val_p,name)
        event('red_entrenada',dict(nombre=name,avisos=net['advertencias']))
    if not all_policies:raise RuntimeError('No hay corrector convergente para seleccionar.')
    selected=choose_policy(all_policies,val_baseline)
    experimental=max(all_policies,key=lambda a:(a['exactitud_balanceada'],-a['FP'],-a['FN']))
    operational=selected or dict(red=experimental['red'],umbral_negativos=1.01,umbral_positivos=0.,**val_baseline)
    decision=dict(adoptar=selected is not None,politica=operational,experimental=experimental,
        criterio='Validación: FP y FN no aumentan; al menos uno disminuye; mejora de exactitud balanceada >=0.001. Prueba no participa.',
        baseline=val_baseline)
    # Congelar selección antes de consultar etiquetas de prueba.
    (OUT/'seleccion_antes_de_prueba.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    g.csv(pd.DataFrame(all_policies),OUT/'politicas_validacion.csv')
    if exchange:exchange.validation_ready(decision)
    event('seleccion_congelada',decision)
    train_table=pd.DataFrame(dict(fila_excel=d.loc[tr,'fila_excel'],fold=fold_id,probabilidad_OOF=oof,
        decision_OOF=oof>=threshold,objetivo_error=code,presencia_real=train_y))
    g.csv(train_table,OUT/'objetivos_OOF.csv')
    comparison=[];predictions=[];cm={};class_reports=[];active={};new_errors=[]
    for s,mask in [('validacion',va),('prueba',te)]:
        family={}
        for name in set([operational['red'],experimental['red']]):
            family[name]=conditional_prob(nets[name],meta_all[mask],base[mask])
        regular_p,regular_codes=family[operational['red']];exp_p,_=family[experimental['red']]
        final=apply_policy(base[mask],regular_p,operational);trial=apply_policy(base[mask],exp_p,experimental)
        active[s]=final
        for variant,pred in [('bosque_clima_cultivo',base[mask]),('corrector_recomendado',final),('corrector_experimental',trial)]:
            comparison.append(dict(conjunto=s,variante=variant,**g.stats(yb[mask],pred)))
        # Estado de error inferido, siempre compatible con decisión del bosque.
        inferred=np.where(base[mask],np.where(final,1,2),np.where(final,-1,0))
        true=truth_codes(yb[mask],base[mask])
        cm[s]=confusion_matrix(true,inferred,labels=CODES).tolist()
        for k in CODES:
            selected_k=inferred==k;actual=true==k
            class_reports.append(dict(conjunto=s,codigo=k,significado=MEANING[str(k)],reales=int(actual.sum()),predichos=int(selected_k.sum()),correctos=int((actual&selected_k).sum())))
        changed=base[mask]!=final
        new_errors.append(dict(conjunto=s,
            falsos_negativos_recuperados=int(((~base[mask])&final&yb[mask]).sum()),
            verdaderos_negativos_convertidos_en_falsa_alarma=int(((~base[mask])&final&(~yb[mask])).sum()),
            falsas_alarmas_eliminadas=int((base[mask]&(~final)&(~yb[mask])).sum()),
            positivos_reales_eliminados_por_error=int((base[mask]&(~final)&yb[mask]).sum())))
        table=d.loc[mask,['fila_excel','Municipio','Parcela','Mes_registrado','Quincena','Coordenada_X','Coordenada_Y',g.TARGET]].copy()
        table['conjunto']=s;table['probabilidad_bosque']=pforest[mask];table['bosque_positivo']=base[mask]
        table['probabilidad_presencia_corregida']=regular_p;table['codigo_error_real']=true;table['codigo_error_predicho']=inferred
        for col,k in enumerate(CODES):table['prob_estado_'+str(k)]=regular_codes[:,col]
        table['presencia_predicha']=final;table['presencia_experimental']=trial;predictions.append(table)
    g.csv(pd.DataFrame(comparison),OUT/'comparacion.csv');g.csv(pd.DataFrame(class_reports),OUT/'estados.csv')
    g.csv(pd.DataFrame(new_errors),OUT/'efectos_correccion.csv')
    # Modelos de magnitud ya ajustados sólo con positivos de train; se conservan
    # sus coeficientes y se vuelve a seleccionar familia por MAE de validación.
    old=joblib.load(g.OUT/'modelo.joblib');quantity={k:old[k] for k in ['bank','blocks','reg_representation','regressors','lookup']}
    fq=g.tabular(d,m,'combinadas');z=quantity['reg_representation'].transform(fq);reg_rows=[]
    for name,model in quantity['regressors'].items():
        for s,mask in [('validacion',va),('prueba',te)]:
            q=g.reg_predict(model,name,z[mask],d[mask]);values=np.where(active[s],q,0)
            reg_rows.append(dict(modelo=name,conjunto=s,MAE=float(mean_absolute_error(y[mask],values)),RMSE=float(np.sqrt(mean_squared_error(y[mask],values)))))
    chosen=min([r for r in reg_rows if r['conjunto']=='validacion'],key=lambda v:v['MAE'])['modelo']
    quantity['selected_regression']=chosen
    for table in predictions:
        s=table.conjunto.iloc[0];mask=split==s;q=g.reg_predict(quantity['regressors'][chosen],chosen,z[mask],d[mask]);table['incidencia_predicha']=np.where(active[s],q,0)
    g.csv(pd.concat(predictions),OUT/'predicciones.csv');g.csv(pd.DataFrame(reg_rows),OUT/'siete_regresiones.csv')
    bundle=dict(forest=full_forest,forest_threshold=threshold,forest_features=FOREST_FEATURES,
        correctors=nets,policy=operational,experimental_policy=experimental,quantity=quantity,
        codigos=MEANING,selection=decision)
    g.save_model(bundle,OUT/'modelo.joblib')
    report=dict(diseno='Mismas coordenadas50/20/30 del experimento de redes. Cinco folds OOF por coordenadas dentro de train; umbral bosque elegido sólo con OOF train.',
        forest_features=FOREST_FEATURES,meta_features=list(meta_oof),arquitectura=[6,9,12],codigos=MEANING,
        conteos_objetivos_train={str(k):int((code==k).sum()) for k in CODES},umbral_bosque=threshold,
        seleccion=decision,comparacion=comparison,efectos=new_errors,matrices_estados=cm,orden_estados=CODES,
        regresiones=reg_rows,regresion_elegida=chosen,advertencias_redes={k:v['advertencias'] for k,v in nets.items()},
        pendientes=['Ceros de muestreos17–24 sin confirmar; unidad de incidencia pendiente.',
            'Prueba histórica consultada varias veces: resultados exploratorios, no validación independiente de futuro.',
            'Comunidades dinámicas por periodo: no extrapolar a fechas fuera del año disponible.',
            'Redes ajustadas con covariables de todo train sin incidencia; objetivos supervisados provienen de OOF.',
            'Las probabilidades neuronales no tienen calibración independiente: se usan como puntuaciones de decisión.',
            'Codificar errores en cuatro categorías reorganiza el problema; no aporta una medición nueva de la enfermedad.',
            'Coeficientes de las siete regresiones reutilizados del entrenamiento positivo previo; sólo se reevalúa la puerta de presencia y la elección de familia.'])
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    make_report(report);plot(report);event('finalizado',dict(adoptado=decision['adoptar'],modelo=chosen))
    return report

def predict(bundle,d,variant='recomendado'):
    d=g.enrich(bundle['quantity'],d).reset_index(drop=True)
    p=bundle['forest'].predict_proba(d[FOREST_FEATURES])[:,1];base=p>=bundle['forest_threshold']
    m=g.graph_transform(d,bundle['quantity']['bank'],bundle['quantity']['blocks'])
    policy=bundle['experimental_policy'] if variant=='experimental' else bundle['policy']
    f=meta_features(m,p,bundle['forest_threshold']);prob,states=conditional_prob(bundle['correctors'][policy['red']],f,base)
    active=base if variant=='bosque' else apply_policy(base,prob,policy)
    code=np.where(base,np.where(active,1,2),np.where(active,-1,0))
    z=bundle['quantity']['reg_representation'].transform(g.tabular(d,m,'combinadas'))
    name=bundle['quantity']['selected_regression'];q=np.where(active,g.reg_predict(bundle['quantity']['regressors'][name],name,z,d),0)
    return pd.DataFrame(dict(probabilidad_presencia_bosque=p,puntuacion_presencia_corregida=prob,
        codigo_error_estimado=code,presencia_predicha=active,grupo_final=m.comunidad_fusion,
        incidencia_predicha=q,estado=np.where(active,'cantidad_estimada','sin_roya')))

def make_report(r):
    lines=['# Bosque clima/cultivo y corrector neuronal de cuatro estados','',
      '## Resultado','',pd.DataFrame(r['comparacion']).to_markdown(index=False),'',
      '**Corrección adoptada en validación:** '+str(r['seleccion']['adoptar'])+'. La prueba se consulta después de congelar la elección.','',
      r['seleccion']['criterio'],'','## Significado de los códigos','',
      '| Código | Estado real respecto al bosque |\n|---|---|\n|0|Sin roya, clasificado negativo: TN|\n|−1|Con roya, clasificado negativo: FN|\n|1|Con roya, clasificado positivo: TP|\n|2|Sin roya, clasificado positivo: FP|','',
      'Los números son categorías nominales. El objetivo se construye con incidencia real sólo en entrenamiento/evaluación. Durante inferencia se estima el estado; nunca se utiliza el estado real como entrada.','',
      'Bosque: seis variables climáticas reales y seis etapas de cultivo, sin coordenadas, altitud, comunidades, mes ni Planta. Corrector: comunidades ambientales/ubicación/urbanística/fusión codificadas one-hot, indicador de ventana incompleta, probabilidad y decisión del bosque.','',
      'El corrector tiene tres capas ocultas6–9–12 y salida de cuatro clases. Se compararon ponderaciones de clases0,0.5,1, con arquitectura fija. Estados incompatibles con la decisión conocida del bosque reciben probabilidad0, y se renormalizan los estados posibles.','',
      'La decisión operativa usa dos umbrales de validación, uno para cada rama. Por ello el código operativo no es necesariamente el argmax de las cuatro probabilidades.','',
      r['diseno'],'','## Qué cambios produce','',pd.DataFrame(r['efectos']).to_markdown(index=False),'',
      '## Magnitud','',pd.DataFrame(r['regresiones']).to_markdown(index=False),'',
      'Familia elegida por validación: **'+r['regresion_elegida']+'**. Se reutilizan los coeficientes ajustados previamente sólo sobre positivos train.','',
      '## Aprendizaje y orquestación','',
      'El orquestador coordina roles computacionales: datos, modelación, estadística, auditoría y memoria. Intercambian propuestas, resultados y objeciones en mensajes.jsonl. La memoria guarda evidencia, decisiones y restricciones para la próxima propuesta; no son agentes LLM ni especialistas humanos. El aprendizaje de pesos ocurre al entrenar los modelos.','',
      'La prueba no dirige decisiones automáticas posteriores; al detectar la misma huella de datos, la memoria propone verificar y obtener una cohorte nueva en vez de repetir la búsqueda.','',
      '## Límites','']+['- '+v for v in r['pendientes']]+['',
      'Referencia técnica: predicciones fuera de muestra para entrenar una segunda etapa, https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.StackingClassifier.html ; probabilidades multiclase https://scikit-learn.org/1.8/modules/generated/sklearn.neural_network.MLPClassifier.html']
    if r.get('comparacion_antecedentes_misma_particion'):
        lines+=['','## Comparación con antecedentes en las mismas ubicaciones','',pd.DataFrame(r['comparacion_antecedentes_misma_particion']).to_markdown(index=False),'',
          'El corrector reduce ambos errores frente al nuevo bosque de doce variables. Frente a la cascada anterior, disminuyen las falsas alarmas y aumentan algunas omisiones. No hay una mejora uniforme frente a todas las configuraciones. El bosque previo de variables completas conserva mejor equilibrio global, con menor sensibilidad.']
    if r.get('magnitud_bosque_vs_corrector'):
        lines+=['','## Efecto final sobre magnitud con la misma regresión','',pd.DataFrame(r['magnitud_bosque_vs_corrector']).to_markdown(index=False),'']
    if r.get('historial_aprendizaje'):
        lines+=['','## Memoria acumulada','',
          'Se incorporan '+str(r['historial_aprendizaje']['informes'])+' informes anteriores con diseño, resultados, decisiones y huella de origen; no se agregan métricas de particiones distintas.','',
          'Las lecciones operativas afectan las restricciones y la siguiente propuesta. En esta ejecución la red sin sobreponderar clases ganó en validación. Con datos nuevos se propone explorar ponderación0 y0.25; con la misma huella se propone verificar sin volver a seleccionar sobre la prueba.','',
          'Consulte `aprendizaje_roya/inventario_experimentos.csv`, `memoria.json`, `mensajes.jsonl` y `siguiente_plan.json`.']
    (OUT/'INFORME.md').write_text('\n'.join(lines))

def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[v for v in r['comparacion'] if v['conjunto']=='prueba'];fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,v in zip(axes,rows):
        a=np.array([[v['TN'],v['FP']],[v['FN'],v['TP']]]);ax.imshow(a,cmap='Blues');ax.set_title(v['variante'].replace('_',' '),fontsize=10)
        ax.set_xticks([0,1],['Sin roya','Con roya']);ax.set_yticks([0,1],['Sin roya','Con roya']);ax.set_xlabel('Predicción');ax.set_ylabel('Registro observado')
        for (i,j),n in np.ndenumerate(a):ax.text(j,i,str(n),ha='center',va='center',fontsize=13,color='white' if n>a.max()/2 else 'black')
    fig.suptitle('Prueba por ubicaciones · bosque clima/cultivo y corrector de errores')
    fig.savefig(OUT/'matrices_confusion.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    from corrector_cuatro_estados import run as canonical_run
    canonical_run()
