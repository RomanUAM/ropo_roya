"""Control: sustituir los3RF por3logísticas, mismas entradas y BNN3–8–4.

No mezcla particiones ni reutiliza los pesos del corrector de los bosques.
"""
import json,warnings
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_auc_score,average_precision_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
import roya_humedad_bnn as h

g=h.g;c=h.c;b=h.b;ROOT=h.ROOT;OUT=ROOT/'evaluacion_logisticas_bnn'
NAMES=['contexto','previos24_HR','individual']

def logistic(view):
    pre=OneHotEncoder(handle_unknown='ignore',sparse_output=True) if view==0 else make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    return make_pipeline(pre,LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=3000,tol=1e-6,random_state=42))

def fit_checked(model,x,y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(x,y)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('Regresión logística no convergente: '+str(caught[0].message))
    return model

def metrics(y,decision):
    return dict(**g.stats(y,decision),exactitud=float(accuracy_score(y,decision)),precision=float(precision_score(y,decision,zero_division=0)),
        F1=float(f1_score(y,decision,zero_division=0)),errores=int(np.sum(y!=decision)))

def run(exchange=None):
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    d=g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv');m=pd.read_csv(g.OUT/'comunidades_por_observacion.csv')
    assert np.array_equal(d.fila_excel,m.fila_excel)
    split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';y=d[g.TARGET].to_numpy(float);yb=y>0
    rf_bundle=joblib.load(h.OUT/'modelo.joblib');q=rf_bundle['quantity'].copy();q['regressors']={name:q['regressors'][name] for name in b.REGRESSIONS}
    m=b.extra_transform(d,m,rf_bundle['extra_graphs']);w=h.features(d,q['bank'],rf_bundle['humidity_history'])
    inputs=[m[b.CONTEXT],w,d[c.FOREST_FEATURES]];previous=pd.read_csv(h.OUT/'entradas_OOF.csv',float_precision='round_trip')
    assert np.array_equal(previous.fila_excel,d.loc[tr,'fila_excel']);fold=previous.fold.to_numpy();sites=g.site_ids(d[tr]).to_numpy()
    oof=np.zeros((tr.sum(),3));prob=np.zeros((len(d),3));models=[];iterations=[]
    for j,name in enumerate(NAMES):
        for k in np.unique(fold):
            a=fold!=k;z=fold==k;assert not set(sites[a])&set(sites[z])
            model=fit_checked(logistic(j),inputs[j][tr].iloc[a],yb[tr][a]);oof[z,j]=model.predict_proba(inputs[j][tr].iloc[z])[:,1]
            iterations.append(dict(vista=name,fold=int(k),iteraciones=int(model[-1].n_iter_.max())));print('logistica_OOF',name,int(k),flush=True)
        model=fit_checked(logistic(j),inputs[j][tr],yb[tr]);models.append(model);prob[:,j]=model.predict_proba(inputs[j])[:,1]
        iterations.append(dict(vista=name,fold='ajuste_train',iteraciones=int(model[-1].n_iter_.max())))
    nets=h.VariationalStateNN().fit(oof,yb[tr]);candidates=[]
    for epoch,net in nets.items():
        p=net.predict_details(prob[va]).probabilidad_roya.to_numpy();quality=b.prior.probability_stats(yb[va],p)
        for row in b.prior.score_policies(yb[va],p):candidates.append(dict(epoch=epoch,**row,**quality))
    chosen=min(candidates,key=lambda v:(v['FP']+v['FN'],v['Brier'],-v['exactitud_balanceada']));net=nets[chosen['epoch']]
    selection=dict(red=chosen,criterio='Mínimo FP+FN validación, igual costo; empate por Brier. Mismos checkpoints80/160/320 y cortes0.025..0.975 que RF+BNN.',
        logistica=dict(C=1.,penalizacion='L2',solver='lbfgs',class_weight='balanced',max_iter=3000,tol=1e-6),
        entrada_red='Sólo3probabilidades OOF; corrector reentrenado, misma arquitectura3–8–4.')
    g.csv(pd.DataFrame(candidates),OUT/'candidatos_validacion.csv');(OUT/'seleccion_antes_prueba.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    if exchange:exchange.validation_ready(selection)
    print('seleccion_congelada',selection,flush=True)
    details=net.predict_details(prob);p=details.probabilidad_roya.to_numpy();decision=p>=chosen['umbral'];reference=h.initial_decision(prob)
    rf=pd.read_csv(h.OUT/'predicciones.csv',float_precision='round_trip');assert np.array_equal(rf.fila_excel,d.fila_excel)
    rf_decision=rf.presencia_predicha.to_numpy(bool);rf_prob=rf.probabilidad_roya.to_numpy();rf_reference=rf.referencia_mayoria.to_numpy(bool)
    comparison=[];quality=[];reliability=[]
    for tag,mask in [('validacion',va),('prueba',te)]:
        for name,a in [('RF_mayoria',rf_reference),('RF_BNN',rf_decision),('LR_mayoria',reference),('LR_BNN',decision)]:
            comparison.append(dict(conjunto=tag,modelo=name,**metrics(yb[mask],a[mask])))
        for j,name in enumerate(NAMES):comparison.append(dict(conjunto=tag,modelo='LR_'+name,**metrics(yb[mask],prob[mask,j]>=.5)))
        for name,score in [('RF_BNN',rf_prob),('LR_BNN',p)]:
            quality.append(dict(conjunto=tag,modelo=name,**b.prior.probability_stats(yb[mask],score[mask]),AUC_ROC=float(roc_auc_score(yb[mask],score[mask])),AP=float(average_precision_score(yb[mask],score[mask]))))
            bins=np.minimum((score[mask]*10).astype(int),9)
            for k in np.unique(bins):
                z=bins==k;reliability.append(dict(conjunto=tag,modelo=name,bin=int(k),n=int(z.sum()),probabilidad_media=float(score[mask][z].mean()),fraccion_roya=float(yb[mask][z].mean())))
    z=q['reg_representation'].transform(g.tabular(d,m,'combinadas'));reg_rows=[];quantities={}
    # Selección de familia sólo en validación; mismos coeficientes positivos y ACP.
    for name in b.REGRESSIONS:
        quantities[name]=np.where(decision,g.reg_predict(q['regressors'][name],name,z,d),0)
        for tag,mask in [('validacion',va),('prueba',te)]:
            for system,pred in [('LR_BNN',quantities[name]),('RF_BNN',rf['incidencia_'+name].to_numpy())]:
                reg_rows.append(dict(sistema=system,modelo=name,conjunto=tag,MAE=float(mean_absolute_error(y[mask],pred[mask])),RMSE=float(np.sqrt(mean_squared_error(y[mask],pred[mask])))))
    reg_chosen=min([v for v in reg_rows if v['conjunto']=='validacion' and v['sistema']=='LR_BNN'],key=lambda v:v['MAE'])['modelo']
    bundle=dict(models=models,quantity=q,extra_graphs=rf_bundle['extra_graphs'],humidity_history=rf_bundle['humidity_history'],
        window_features=w.columns.tolist(),bnn=net,selection_HR=selection,selected_regression=reg_chosen,
        humidity_available=True,humidity_simulated_available=False,humidity_source='observada_quincenal',reference_rule='mayoría de3 puntuaciones>=0.5',base_classifier='logistica')
    g.save_model(bundle,OUT/'modelo.joblib')
    rows=pd.concat([d[['fila_excel','Municipio','Parcela','Coordenada_X','Coordenada_Y','Mes_registrado','Quincena',g.TARGET]].reset_index(drop=True),details,b.outcomes(p,decision)],axis=1)
    rows['conjunto']=split;rows['presencia_predicha']=decision;rows['referencia_mayoria']=reference;rows['codigo_real_referencia']=c.truth_codes(yb,reference)
    rows['P_error_final']=np.where(decision,1-p,p);rows['P_correcto_final']=1-rows.P_error_final
    rows['requiere_revision']=(rows.posterior_inferior95<=chosen['umbral'])&(rows.posterior_superior95>=chosen['umbral'])
    for j,name in enumerate(NAMES):rows['p_'+name]=prob[:,j]
    for name,value in quantities.items():rows['incidencia_'+name]=value
    rows['incidencia_predicha']=quantities[reg_chosen];g.csv(rows,OUT/'predicciones.csv')
    o=pd.DataFrame(oof,columns=['p_'+name for name in NAMES]);o['fila_excel']=d.loc[tr,'fila_excel'].to_numpy();o['fold']=fold;o['presencia_real']=yb[tr];o['referencia_mayoria']=h.initial_decision(oof);o['codigo_real_referencia']=c.truth_codes(yb[tr],h.initial_decision(oof));g.csv(o,OUT/'entradas_OOF.csv')
    g.csv(pd.DataFrame(comparison),OUT/'comparacion.csv');g.csv(pd.DataFrame(quality),OUT/'calidad_probabilidades.csv');g.csv(pd.DataFrame(reliability),OUT/'calibracion.csv');g.csv(pd.DataFrame(reg_rows),OUT/'dos_regresiones.csv');g.csv(pd.DataFrame(iterations),OUT/'convergencia_logisticas.csv');g.csv(pd.DataFrame(net.trace),OUT/'entrenamiento_BNN.csv')
    subgroups=[]
    for municipio,ix in d.loc[te].groupby('Municipio').groups.items():
        for name,a in [('RF_BNN',rf_decision),('LR_BNN',decision)]:subgroups.append(dict(Municipio=municipio,modelo=name,n=len(ix),**metrics(yb[ix],a[ix])))
    g.csv(pd.DataFrame(subgroups),OUT/'municipios.csv')
    bootstrap=paired_bootstrap(d.loc[te].reset_index(drop=True),yb[te],rf_decision[te],decision[te]);g.csv(pd.DataFrame(bootstrap),OUT/'bootstrap_pareado_por_sitio.csv')
    report=dict(diseno='Mismos2200sitios:1100train,440validación,660prueba;28728/11520/17352registros. Se conservan5folds OOF por coordenadas y se reentrenan3logísticas y su BNN.',
        seleccion=selection,entradas=dict(contexto=b.CONTEXT,antecedentes=w.columns.tolist(),individual=c.FOREST_FEATURES),
        comparacion=comparison,calidad_probabilidades=quality,regresiones=reg_rows,regresion_elegida=reg_chosen,municipios=subgroups,bootstrap=bootstrap,
        comparabilidad='Mismas entradas, comunidades, historiaHR, división, arquitectura y presupuesto de selección de BNN. Cambia sólo la familia de clasificadores base y sus parámetros propios. La BNN se reentrena para cada familia.',
        clasificacion_elegida_validacion=min([v for v in comparison if v['conjunto']=='validacion' and v['modelo'] in ['RF_BNN','LR_BNN']],key=lambda v:(v['errores'],-v['exactitud_balanceada']))['modelo'],
        limites=['Prueba histórica reutilizada: comparación exploratoria, no validación prospectiva.',
            'Muestreos 17-24 en cero y unidad de incidencia aún por verificar.',
            'HR real quincenal; 75/80% son umbrales exploratorios. No hay humedad simulada diaria ni mojado foliar.',
            'Rezagos 20-24: hipótesis ambientales, no colonización observada ni una latencia universal.',
            'Comunidades nominales de grafos ajustados con covariables train; dentro de OOF conservan representación transductiva sin etiquetas de enfermedad.',
            'Las ventanas usan día 1/16 por falta de fecha exacta. Inferencia de fechas nuevas exige actualizar las comunidades.',
            'Intervalos variacionales aproximados sobre pesos; no cobertura epidemiológica.',
            'Bootstrap pareado por sitio y estratificado por municipio, condicionado a esta cohorte; no corrige consultas previas a prueba.',
            'Se comparan dos arquitecturas actuales; no es un ranking de todos los perfiles históricos.'])
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));write_report(report);plot(report);return report

def paired_bootstrap(d,y,rf,lr):
    frame=pd.DataFrame(dict(sitio=g.site_ids(d),municipio=d.Municipio,delta_FP=(lr&~y).astype(int)-(rf&~y).astype(int),delta_FN=(~lr&y).astype(int)-(~rf&y).astype(int)))
    agg=frame.groupby(['municipio','sitio'])[['delta_FP','delta_FN']].sum();rng=np.random.default_rng(42);draws=[]
    for _ in range(1000):
        result=np.zeros(2)
        for _,block in agg.groupby(level=0):
            x=block.to_numpy();result+=x[rng.integers(0,len(x),len(x))].sum(axis=0)
        draws.append(result)
    draws=np.asarray(draws);rows=[]
    for name,point,values in [('FP',agg.delta_FP.sum(),draws[:,0]),('FN',agg.delta_FN.sum(),draws[:,1]),('FP+FN',agg.sum().sum(),draws.sum(axis=1))]:
        rows.append(dict(metrica=name,diferencia_LR_menos_RF=int(point),inferior95=float(np.quantile(values,.025)),superior95=float(np.quantile(values,.975)),replicas=1000))
    return rows

def predict(bundle,d):return h.predict(bundle,d)

def write_report(r):
    main=pd.DataFrame(r['comparacion']);final=main[main.modelo.isin(['RF_BNN','LR_BNN'])]
    lines=['# Comparación completa: tres bosques o tres logísticas + BNN','',r['diseno'],'',r['comparabilidad'],'',
        '## Arquitecturas','',
        'RF: tres Random Forest (contexto, antecedentes térmicos24d+HR, clima real+fenología) → BNN3–8–4 → decisión → dos regresiones de cantidad.\n\nLR: tres logísticas con exactamente los mismos bloques → BNN3–8–4 reentrenada → decisión → las mismas dos regresiones.',
        '', 'Las comunidades se codifican one-hot; las logísticas numéricas usan mediana y StandardScaler ajustados dentro de cada fold. Logísticas L2,C=1,pesos balanceados,lbfgs,tolerancia1e-6y máximo3000iteraciones. Todas deben converger. No se hace búsqueda adicional deC ni de arquitecturas.',
        '', 'La BNN conserva Bayes by Backprop, posterior Gaussiano diagonal en pesos/sesgos, priorN(0,1),tanh y128sorteos predictivos. La mayoría inicial usa0.5en cada clasificador. Se entrenan cuatro estadosFN/TN/TP/FP relativos a esa mayoría, yP(roya)=P(FN)+P(TP). Arquitectura3–8–4 fija; mismo Adam, semilla y checkpoints80/160/320. Se escogen checkpoint y corte por mínimoFP+FNde validación, con Brier para desempatar. No se usan etiquetas de prueba para seleccionar.',
        '', '## Resultados finales','',final.to_markdown(index=False),'',
        'Filas de cada matriz: estado real. Columnas: predicción. Orden sin roya/con roya; celdas[[TN,FP],[FN,TP]].',
        '', '## Clasificadores y efecto de la red','',main.to_markdown(index=False),'',
        '## Calidad de probabilidades','',pd.DataFrame(r['calidad_probabilidades']).to_markdown(index=False),'',
        '## Dos regresiones de cantidad','',pd.DataFrame(r['regresiones']).to_markdown(index=False),'',
        'Coeficientes de regresión y ACP positivos previos conservados para aislar el efecto de la puerta de presencia. Familia elegida por MAE de validación paraLR: '+r['regresion_elegida']+'. Clasificador preferido entre las dos rutas por errores de validación: '+r['clasificacion_elegida_validacion']+'.',
        '', '## Comparación por municipio','',pd.DataFrame(r['municipios']).to_markdown(index=False),'',
        '## Incertidumbre de diferencias','',pd.DataFrame(r['bootstrap']).to_markdown(index=False),'',
        'Diferencias LR menosRF: negativo favoreceLR. Intervalos percentiles de1000remuestreos pareados de sitios, dentro de cada municipio. Son exploratorios y no justifican retocar el corte con prueba.',
        '', '## Límites','']+['- '+s for s in r['limites']]+['',
        '## Próximas mejoras','',
        'Validar otra cohorte antes de buscar nuevos hiperparámetros. Evaluar grafos reconstruidos dentro de cada fold; completarHR diaria/mojado foliar, verificar ceros y fijar el costo relativo de omitir roya e inspeccionar una falsa alarma. No añadir complejidad sólo porque exista otra arquitectura disponible.']
    (OUT/'INFORME.md').write_text('\n'.join(lines))

def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for tag in ['validacion','prueba']:
        rows=[v for v in r['comparacion'] if v['conjunto']==tag and v['modelo'] in ['RF_BNN','LR_BNN']]
        fig,axes=plt.subplots(1,2,figsize=(9.5,4.5),constrained_layout=True);vmax=max(max(v[k] for k in ['TN','FP','FN','TP']) for v in rows)
        for ax,v in zip(axes,rows):
            a=np.array([[v['TN'],v['FP']],[v['FN'],v['TP']]]);ax.imshow(a,cmap='Blues',vmin=0,vmax=vmax)
            ax.set_title(('Bosques' if v['modelo']=='RF_BNN' else 'Logísticas')+' + red bayesiana',fontsize=12)
            ax.set_xticks([0,1],['Sin roya','Con roya']);ax.set_yticks([0,1],['Sin roya','Con roya']);ax.set_xlabel('Predicción');ax.set_ylabel('Estado real')
            for (i,j),n in np.ndenumerate(a):ax.text(j,i,f'{n:,}',ha='center',va='center',fontsize=18,color='white' if n>vmax/2 else '#12243a')
        fig.suptitle('Matrices de confusión · '+tag+' · mismas ubicaciones',fontsize=14);fig.savefig(OUT/('matrices_'+tag+'.png'),dpi=180);plt.close(fig)

if __name__=='__main__':
    from comparar_logisticas_bnn import run as canonical_run
    canonical_run()
