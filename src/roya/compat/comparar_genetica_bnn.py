"""Tres GP probabilísticas + BNN, protocolo común con RF y logísticas."""
import json,hashlib,itertools
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from genetica_probabilistica3 import GeneticProbability3
from programacion_genetica_roya import levels
import comparar_logisticas_bnn as common

h=common.h;g=h.g;b=h.b;c=h.c;ROOT=h.ROOT;OUT=ROOT/'evaluacion_genetica_bnn';NAMES=common.NAMES

def genetic(view,seed=42):
    pre=OneHotEncoder(handle_unknown='ignore',sparse_output=False) if view==0 else make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    return make_pipeline(pre,GeneticProbability3(population=256,generations=60,seed=seed,parsimony=.0005))

def reference_bundle():
    return joblib.load(h.OUT/'modelo.joblib')

def resolve_bundle(light):
    path=ROOT/light['reference_relative_path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=light['reference_sha256']:
        raise ValueError('Cambió el modelo de referencia compartido: verificar o reentrenar GP antes de predecir.')
    ref=joblib.load(path)
    return dict(models=light['models'],quantity=ref['quantity'],extra_graphs=ref['extra_graphs'],humidity_history=ref['humidity_history'],
        bnn=light['bnn'],selection_HR=light['selection_HR'],selected_regression=light['selected_regression'])

def predict(light,d):return h.predict(resolve_bundle(light),d)

def run(exchange=None):
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    d=g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv');m=pd.read_csv(g.OUT/'comunidades_por_observacion.csv')
    assert np.array_equal(d.fila_excel,m.fila_excel)
    split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';y=d[g.TARGET].to_numpy(float);yb=y>0
    ref=reference_bundle();q=ref['quantity'];m=b.extra_transform(d,m,ref['extra_graphs']);w=h.features(d,q['bank'],ref['humidity_history'])
    inputs=[m[b.CONTEXT],w,d[c.FOREST_FEATURES]];old_oof=pd.read_csv(h.OUT/'entradas_OOF.csv',float_precision='round_trip');lr_oof=pd.read_csv(common.OUT/'entradas_OOF.csv',float_precision='round_trip')
    assert np.array_equal(old_oof.fila_excel,lr_oof.fila_excel) and np.array_equal(old_oof.fold,lr_oof.fold)
    assert np.array_equal(old_oof.fila_excel,d.loc[tr,'fila_excel']);fold=old_oof.fold.to_numpy();sites=g.site_ids(d[tr]).to_numpy()
    shared=dict(particion='50/20/30por coordenadas',sitios=[1100,440,660],registros=[int(tr.sum()),int(va.sum()),int(te.sum())],
        objetivo='Incidencia de roya del Café >0',positivos_y_negativos=True,
        folds_OOF=5,sha256_filas_folds=hashlib.sha256(old_oof[['fila_excel','fold']].to_csv(index=False).encode()).hexdigest(),
        entradas=dict(zip(NAMES,[list(x) for x in inputs])),
        representacion='Comunidades nominales one-hot; imputación por mediana fit-fold. Numéricas escaladas para LR/GP; árboles RF no necesitan escalado.',
        balance_base=dict(RF='balanced_subsample',LR='balanced',GP='n/(2*n_clase) dentro de cada fold'),
        ajustes_base='Una configuración fija por familia; funciones de pérdida propias de cada algoritmo, sin búsqueda adicional en GP.',
        BNN=dict(arquitectura=[3,8,4],semilla=42,epochs=320,checkpoints=[80,160,320],batch_maximo=512,Adam_lr=.005,
            prior='Normal(0,1)',posterior='Gaussiano diagonal variacional',muestras=128,referencia_inicial='Mayoría con corte0.5',
            pesos_objetivo='No ponderados; minimizar errores finales con igual costo'),
        seleccion=dict(metrica='FP+FN en validación',desempate='Brier, después exactitud balanceada',cortes='0.025..0.975 paso0.025',prueba_no_selecciona=True),
        regresiones=dict(modelos=b.REGRESSIONS,coeficientes_y_ACP='Los mismos, positivos train originales',seleccion='MAE validación'),
        incertidumbre='Bootstrap pareado por sitio estratificado por municipio,1000réplicas,semilla42')
    (OUT/'protocolo_comun.json').write_text(json.dumps(shared,ensure_ascii=False,indent=2))
    oof=np.zeros((tr.sum(),3));prob=np.zeros((len(d),3));models=[];histories=[];expressions=[];variables=[]
    for j,name in enumerate(NAMES):
        for k in np.unique(fold):
            a=fold!=k;z=fold==k;assert not set(sites[a])&set(sites[z])
            model=genetic(j,42+int(k)).fit(inputs[j][tr].iloc[a],yb[tr][a]);oof[z,j]=model.predict_proba(inputs[j][tr].iloc[z])[:,1]
            gp=model[-1];histories.extend(dict(vista=name,fold=int(k),**v) for v in gp.history_)
            expressions.append(dict(vista=name,fold=int(k),expresion=gp.expression_,niveles=levels(gp.tree_),programas_evaluados=gp.programs_evaluated_,fitness=gp.fitness_))
            print('GP_OOF',name,int(k),gp.expression_,flush=True)
        model=genetic(j).fit(inputs[j][tr],yb[tr]);models.append(model);prob[:,j]=model.predict_proba(inputs[j])[:,1];gp=model[-1]
        histories.extend(dict(vista=name,fold='ajuste_train',**v) for v in gp.history_)
        expressions.append(dict(vista=name,fold='ajuste_train',expresion=gp.expression_,niveles=levels(gp.tree_),programas_evaluados=gp.programs_evaluated_,fitness=gp.fitness_))
        encoded=model[0].get_feature_names_out()
        variables.extend(dict(vista=name,terminal='Z'+str(i+1),variable=str(v)) for i,v in enumerate(encoded))
        print('GP_train',name,gp.expression_,flush=True)
    nets=h.VariationalStateNN().fit(oof,yb[tr]);candidates=[]
    for epoch,net in nets.items():
        p=net.predict_details(prob[va]).probabilidad_roya.to_numpy();quality=b.prior.probability_stats(yb[va],p)
        for row in b.prior.score_policies(yb[va],p):candidates.append(dict(epoch=epoch,**row,**quality))
    chosen=min(candidates,key=lambda v:(v['FP']+v['FN'],v['Brier'],-v['exactitud_balanceada']));net=nets[chosen['epoch']]
    validation_competitors=[]
    for source,tag in [(h.OUT,'RF_BNN'),(common.OUT,'LR_BNN')]:
        scores=pd.read_csv(source/'comparacion.csv');row=scores[(scores.conjunto=='validacion')&(scores.modelo==('red_bayesiana_HR' if tag=='RF_BNN' else tag))].iloc[0]
        quality=pd.read_csv(source/'calidad_probabilidades.csv');qr=quality[(quality.conjunto=='validacion')&(quality.modelo==('red_bayesiana_HR' if tag=='RF_BNN' else tag))].iloc[0]
        validation_competitors.append(dict(modelo=tag,errores=int(row.FP+row.FN),Brier=float(qr.Brier),exactitud_balanceada=float(row.exactitud_balanceada)))
    validation_competitors.append(dict(modelo='GP_BNN',errores=chosen['FP']+chosen['FN'],Brier=chosen['Brier'],exactitud_balanceada=chosen['exactitud_balanceada']))
    preferred=min(validation_competitors,key=lambda v:(v['errores'],v['Brier'],-v['exactitud_balanceada']))['modelo']
    selection=dict(red=chosen,criterio=shared['seleccion'],preferencia_comparacion=preferred,comparacion_validacion=validation_competitors,
        GP=dict(poblacion=256,generaciones=60,niveles_maximos=3,parsimonia=.0005,cruce=.7,mutacion=.25,elitismo=8,torneo=5,
            fitness='Log-loss Bernoulli balanceada +0.0005por nodo',salida='sigmoide(programa)'),
        entrada_red='Sólo3probabilidades OOF; misma BNN reentrenada por familia.')
    g.csv(pd.DataFrame(candidates),OUT/'candidatos_validacion.csv');g.csv(pd.DataFrame(expressions),OUT/'expresiones.csv');g.csv(pd.DataFrame(histories),OUT/'evolucion_GP.csv');g.csv(pd.DataFrame(variables),OUT/'diccionario_terminales.csv')
    (OUT/'seleccion_antes_prueba.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    if exchange:exchange.validation_ready(selection)
    print('seleccion_congelada',selection,flush=True)
    details=net.predict_details(prob);p=details.probabilidad_roya.to_numpy();decision=p>=chosen['umbral'];reference=h.initial_decision(prob)
    previous={name:pd.read_csv(source/'predicciones.csv',float_precision='round_trip') for name,source in [('RF_BNN',h.OUT),('LR_BNN',common.OUT)]}
    for frame in previous.values():assert np.array_equal(frame.fila_excel,d.fila_excel)
    scores={name:frame.probabilidad_roya.to_numpy() for name,frame in previous.items()};scores['GP_BNN']=p
    decisions={name:frame.presencia_predicha.to_numpy(bool) for name,frame in previous.items()};decisions['GP_BNN']=decision
    references={name:frame.referencia_mayoria.to_numpy(bool) for name,frame in previous.items()};references['GP_BNN']=reference
    comparison=[];quality=[]
    for tag,mask in [('validacion',va),('prueba',te)]:
        for name,a in decisions.items():
            comparison.append(dict(conjunto=tag,modelo=name,**common.metrics(yb[mask],a[mask])))
            comparison.append(dict(conjunto=tag,modelo=name.replace('_BNN','_mayoria'),**common.metrics(yb[mask],references[name][mask])))
            quality.append(dict(conjunto=tag,modelo=name,**b.prior.probability_stats(yb[mask],scores[name][mask]),AUC_ROC=float(roc_auc_score(yb[mask],scores[name][mask])),AP=float(average_precision_score(yb[mask],scores[name][mask]))))
        for j,name in enumerate(NAMES):comparison.append(dict(conjunto=tag,modelo='GP_'+name,**common.metrics(yb[mask],prob[mask,j]>=.5)))
    z=q['reg_representation'].transform(g.tabular(d,m,'combinadas'));reg_rows=[];quantities={}
    for name in b.REGRESSIONS:
        quantities[name]=np.where(decision,g.reg_predict(q['regressors'][name],name,z,d),0)
        for tag,mask in [('validacion',va),('prueba',te)]:
            for system in ['RF_BNN','LR_BNN','GP_BNN']:
                value=quantities[name] if system=='GP_BNN' else previous[system]['incidencia_'+name].to_numpy()
                reg_rows.append(dict(sistema=system,modelo=name,conjunto=tag,MAE=float(mean_absolute_error(y[mask],value[mask])),RMSE=float(np.sqrt(mean_squared_error(y[mask],value[mask])))))
    reg_chosen=min([v for v in reg_rows if v['conjunto']=='validacion' and v['sistema']=='GP_BNN'],key=lambda v:v['MAE'])['modelo']
    # Compartir el bloque inmutable evita duplicar datos, grafos y regresiones.
    refpath=h.OUT/'modelo.joblib';light=dict(models=models,bnn=net,selection_HR=selection,selected_regression=reg_chosen,
        reference_relative_path=str(refpath.relative_to(ROOT)),reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
        window_features=w.columns.tolist(),base_classifier='programacion_genetica',protocolo_comun=shared)
    g.save_model(light,OUT/'modelo.joblib')
    rows=pd.concat([d[['fila_excel','Municipio','Parcela','Coordenada_X','Coordenada_Y',g.TARGET]].reset_index(drop=True),details,b.outcomes(p,decision)],axis=1)
    rows['conjunto']=split;rows['presencia_predicha']=decision;rows['referencia_mayoria']=reference;rows['codigo_real_referencia']=c.truth_codes(yb,reference)
    rows['P_error_final']=np.where(decision,1-p,p);rows['P_correcto_final']=1-rows.P_error_final
    rows['requiere_revision']=(rows.posterior_inferior95<=chosen['umbral'])&(rows.posterior_superior95>=chosen['umbral'])
    for j,name in enumerate(NAMES):rows['p_'+name]=prob[:,j]
    for name,value in quantities.items():rows['incidencia_'+name]=value
    rows['incidencia_predicha']=quantities[reg_chosen];g.csv(rows,OUT/'predicciones.csv')
    o=pd.DataFrame(oof,columns=['p_'+name for name in NAMES]);o['fila_excel']=d.loc[tr,'fila_excel'].to_numpy();o['fold']=fold;o['presencia_real']=yb[tr];o['referencia_mayoria']=h.initial_decision(oof);o['codigo_real_referencia']=c.truth_codes(yb[tr],h.initial_decision(oof));g.csv(o,OUT/'entradas_OOF.csv')
    g.csv(pd.DataFrame(comparison),OUT/'comparacion.csv');g.csv(pd.DataFrame(quality),OUT/'calidad_probabilidades.csv');g.csv(pd.DataFrame(reg_rows),OUT/'dos_regresiones.csv');g.csv(pd.DataFrame(net.trace),OUT/'entrenamiento_BNN.csv')
    subgroups=[]
    for municipio,ix in d.loc[te].groupby('Municipio').groups.items():
        for name,a in decisions.items():subgroups.append(dict(Municipio=municipio,modelo=name,n=len(ix),**common.metrics(yb[ix],a[ix])))
    g.csv(pd.DataFrame(subgroups),OUT/'municipios.csv');boot=[]
    for candidate,refname in [('LR_BNN','RF_BNN'),('GP_BNN','RF_BNN'),('GP_BNN','LR_BNN')]:
        for row in common.paired_bootstrap(d.loc[te].reset_index(drop=True),yb[te],decisions[refname][te],decisions[candidate][te]):
            row['diferencia_candidato_menos_referencia']=row.pop('diferencia_LR_menos_RF');boot.append(dict(candidato=candidate,referencia=refname,**row))
    g.csv(pd.DataFrame(boot),OUT/'bootstrap_pareado.csv')
    report=dict(diseno='Protocolo común50/20/30por ubicación, mismas filas y5folds OOF; probabilidades de cada familia corregidas por BNN3–8–4 reentrenada.',
        protocolo=shared,seleccion=selection,comparacion=comparison,calidad_probabilidades=quality,regresiones=reg_rows,regresion_elegida=reg_chosen,municipios=subgroups,bootstrap=boot,expresiones=expressions,
        limites=['Prueba histórica reutilizada; resultados exploratorios, sin promoción a producción.',
            'Muestreos 17-24 en cero y unidad de incidencia pendientes de verificar.',
            'Humedad real quincenal; no humedad simulada diaria ni mojado foliar. 75/80% son hipótesis.',
            'Rezagos 20-24 exploratorios; fechas aproximadas 1/16, no colonización observada.',
            'Comunidades ajustadas con covariables train; OOF transductivo en esa representación, sin etiquetas de enfermedad.',
            'GP limitada a 3 niveles y 60 generaciones: se compara esta configuración, no toda programación genética posible.',
            'Sigmoide de una fórmula no garantiza calibración; Brier/log-loss se evalúan por separado.',
            'Intervalos de pesos variacionales y bootstrap por sitio condicionados a esta cohorte; no validación prospectiva.',
            'GP comparte grafos, historia y regresiones inmutables con la referencia RF; se verifica su hash al predecir. El paquete completo contiene ambos.'])
    report['diagnosticos']=dict(GP_contexto_todos_positivos_corte05=bool((prob[te,0]>=.5).all()),
        valores_contexto_prueba=np.unique(prob[te,0]).tolist(),interpretacion='La sigmoide no basta para dar un detector útil al corte0.5; no cambiar el corte sólo paraGP mirando prueba.')
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));write_report(report);plot(report);return report

def write_report(r):
    main=pd.DataFrame(r['comparacion']);final=main[main.modelo.isin(['RF_BNN','LR_BNN','GP_BNN'])]
    lines=['# Tres familias con un protocolo común: RF, logísticas y GP + BNN','',r['diseno'],'',
        'La GP evoluciona una fórmula por vista: P(roya)=sigmoide(programa). Se fijan256individuos,60generaciones y3niveles máximos, pérdida Bernoulli balanceada y penalización0.0005por nodo; una sola configuración. Funciones: suma, resta, producto, división protegida, seno, coseno ylog(1+abs). Las comunidades se codifican one-hot, nunca como números ordinales.','',
        'Las tres familias comparten filas, partición, cinco folds y variables. LR/GP escalan numéricas en train-fold; RF no requiere escalado. Cada algoritmo conserva su pérdida y regularización propias. Esto es mismo protocolo experimental, no algoritmos idénticos.','',
        'La BNN3–8–4 se reentrena para cada familia con misma semilla, Adam, batch, máximo320épocas, checkpoints80/160/320,128sorteos y búsqueda de corte0.025..0.975. Selección por mínimoFP+FN de validación, empate por Brier. Puede elegir checkpoints/cortes distintos bajo la misma regla.','',
        '## Matrices y clasificación final','',final.to_markdown(index=False),'',
        'Filas de matrices=estado real, columnas=predicción. Orden sin/con roya:[[TN,FP],[FN,TP]]. Preferencia fijada en validación: **'+r['seleccion']['preferencia_comparacion']+'**.','',
        '## Antes y después de la red; GP por vista','',main.to_markdown(index=False),'',
        'Diagnóstico: GP contextual clasifica todos los registros de prueba como positivos al corte común0.5. Sus puntuaciones son '+str(r.get('diagnosticos',{}).get('valores_contexto_prueba',[]))+'. Esto limita su uso como detector aislado; no se cambia su corte mirando prueba. La BNN puede reducir falsas alarmas e introducir omisiones: se informa el intercambio completo.','',
        '## Probabilidades','',pd.DataFrame(r['calidad_probabilidades']).to_markdown(index=False),'',
        '## Dos regresiones idénticas tras la puerta','',pd.DataFrame(r['regresiones']).to_markdown(index=False),'',
        'Mismos coeficientes y ACP positivos originales; sólo cambia la puerta de presencia. GP elige por MAE de validación: '+r['regresion_elegida']+'.','',
        '## Municipio y diferencias pareadas','',pd.DataFrame(r['municipios']).to_markdown(index=False),'',pd.DataFrame(r['bootstrap']).to_markdown(index=False),'',
        'Diferencias candidato menos referencia; negativo favorece al candidato.1000remuestreos pareados de sitios, estratificados por municipio. No corrige reutilización de prueba.','',
        '## Fórmulas finales de GP','',pd.DataFrame([v for v in r['expresiones'] if v['fold']=='ajuste_train']).to_markdown(index=False),'',
        'Z1,Z2,... se definen en diccionario_terminales.csv; las numéricas son estandarizadas. La sigmoide se aplica después del programa y no es otra regresión logística ajustada.','',
        '## Límites','']+['- '+s for s in r['limites']]+['',
        'Siguiente paso: nueva cohorte con ceros y positivos verificados,HR diaria/mojado, evaluación temporal y espacial. No ampliar generaciones/capas buscando una mejora sobre esta misma prueba.']
    (OUT/'INFORME.md').write_text('\n'.join(lines))

def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for tag in ['validacion','prueba']:
        rows=[v for v in r['comparacion'] if v['conjunto']==tag and v['modelo'] in ['RF_BNN','LR_BNN','GP_BNN']]
        fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True);vmax=max(max(v[k] for k in ['TN','FP','FN','TP']) for v in rows)
        for ax,v in zip(axes,rows):
            a=np.array([[v['TN'],v['FP']],[v['FN'],v['TP']]]);ax.imshow(a,cmap='Blues',vmin=0,vmax=vmax)
            ax.set_title({'RF_BNN':'Bosques','LR_BNN':'Logísticas','GP_BNN':'Programación genética'}[v['modelo']]+' + BNN',fontsize=16)
            ax.set_xticks([0,1],['Sin roya','Con roya'],fontsize=14);ax.set_yticks([0,1],['Sin roya','Con roya'],fontsize=14);ax.set_xlabel('Predicción',fontsize=14);ax.set_ylabel('Estado real',fontsize=14)
            for (i,j),n in np.ndenumerate(a):ax.text(j,i,f'{n:,}',ha='center',va='center',fontsize=23,color='white' if n>vmax/2 else '#12243a')
        fig.suptitle('Mismo protocolo · matrices de confusión · '+tag,fontsize=18);fig.savefig(OUT/('matrices_'+tag+'.png'),dpi=180);plt.close(fig)

if __name__=='__main__':
    from comparar_genetica_bnn import run as canonical_run
    canonical_run()
