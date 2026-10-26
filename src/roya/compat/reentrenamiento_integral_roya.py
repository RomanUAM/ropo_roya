"""Reentrenamiento común RF/LR/GP, topografía original y6regresiones.

Sólo registros de la base; puerta negativa no ejecuta regresiones.
"""
import json,hashlib,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import estructura_integral_roya as e
import comparar_logisticas_bnn as common

h=e.h;g=e.g;b=e.b;c=e.c;ROOT=h.ROOT;OUT=ROOT/'evaluacion_integral_binaria_topografia';FAMILIES=['RF','LR','GP']
LABELS={'RF':'Bosques','LR':'Logísticas','GP':'Programación genética'}
GEO_SOURCE='geografia_roya/base_geografica_exploratoria.csv';TERRAIN_SOURCE='barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv';REF_SOURCE='evaluacion_humedad_bnn/modelo.joblib'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def assemble():return e.typed_data(g.assemble(ROOT/GEO_SOURCE,ROOT/TERRAIN_SOURCE))

def audit(d):
    counts=d[g.STAGES].sum(axis=1).value_counts().sort_index().to_dict();u=d.assign(sitio=g.site_ids(d)).groupby('sitio')[['Altitud','Pendiente_grad','Orientacion']].nunique()
    bad=u.index[(u>1).any(axis=1)];conflicts=d[g.site_ids(d).isin(bad)][['Municipio','Parcela','Coordenada_X','Coordenada_Y','Altitud','Pendiente_grad','Orientacion']].drop_duplicates()
    g.csv(conflicts,OUT/'conflictos_topograficos_coordenadas.csv')
    report=dict(registros=len(d),fenologia={col:sorted(d[col].unique().tolist()) for col in g.STAGES},
        indicadores_activos_por_registro={str(k):int(v) for k,v in counts.items()},fenologia_multietiqueta=True,
        coordenadas_unicas=int(g.site_ids(d).nunique()),nodos_parcela_coordenada=int(h.keys(d).nunique()),coordenadas_con_metadatos_contradictorios=len(bad),
        tratamiento='Se conserva identidad parcela+coordenada para red topográfica. La partición sigue agrupando por coordenada, sin separar duplicados.',
        orientacion='Circular seno/coseno; PLANO tiene seno=coseno=0 y bandera propia. Se conservan relieveMDE y datos originales por separado.')
    write(OUT/'auditoria_datos.json',report);return report

def load_reference():return joblib.load(ROOT/REF_SOURCE)

def prepare():
    OUT.mkdir(exist_ok=False);d=assemble();data_audit=audit(d);m=pd.read_csv(g.OUT/'comunidades_por_observacion.csv');assert np.array_equal(m.fila_excel,d.fila_excel)
    split=m.conjunto.to_numpy();tr=split=='train';oof=pd.read_csv(h.OUT/'entradas_OOF.csv');assert np.array_equal(oof.fila_excel,d.loc[tr,'fila_excel'])
    fold=oof.fold.to_numpy();ref=load_reference();m=b.extra_transform(d,m,ref['extra_graphs']);w=h.features(d,ref['quantity']['bank'],ref['humidity_history'])
    train_coords=set(g.site_ids(d[tr]));topo,labels,report=e.fit_topography(d,train_coords,OUT,'train');m['comunidad_topografia']=labels
    g.save_model(topo,OUT/'topografia.joblib');g.csv(m,OUT/'comunidades_por_fila.csv');reports=[report];fold_context={};groups=g.site_ids(d[tr]).to_numpy()
    for k in sorted(np.unique(fold)):
        a=fold!=k;z=fold==k;assert not set(groups[a])&set(groups[z])
        _,newlabels,r=e.fit_topography(d,set(groups[a]),None,'OOF_'+str(k));reports.append(r)
        f=m[b.CONTEXT].copy();f['comunidad_topografia']=newlabels;fold_context[int(k)]=f.loc[tr].reset_index(drop=True)
        print('topografia_OOF',int(k),'nodos',r['nodos'],'comunidades',r['comunidades'],flush=True)
    g.csv(pd.DataFrame(reports),OUT/'redes_topografia_resumen.csv')
    protocol=dict(datos='Sólo57,600registros observados; sin filas sintéticas de respuesta.',
        particion='Mismas coordenadas50/20/30:1100/440/660;28728/11520/17352registros.',OOF='Mismos5folds por coordenadas; topografía nueva ajustada dentro de cada fold.',
        topografia=e.topo_columns(),fenologia='Seis indicadores binarios independientes, sin escalado ni ACP en esas columnas; no se fuerza una única etapa.',
        otras_comunidades='Se conservan comunidades geográficas/simuladas comunes ajustadas con covariables train; representación transductiva dentro de OOF, sin etiquetas de enfermedad.',
        bases=dict(RF='180árboles,profundidad12,hoja15,balanced_subsample',LR='L2,C=1,balanced,lbfgs3000',GP='256individuos,60generaciones,3niveles,logloss balanceada+parsimonia0.0005'),
        BNN=dict(arquitectura=[3,8,4],checkpoints=[80,160,320],muestras=128,seed=42,criterio='MínimoFP+FN validación; empateBrier y balanceada',cortes='0.025..0.975paso0.025'),
        regresiones=e.REGRESSIONS,ajuste_cantidad='OLS, sólo positivos reales de train; ACP95% sólo continuas, indicadores binarios y comunidades estáticas aditivos.',
        modelos_armonicos='Periodo365días; armónicos1y2. Sexta familia=polinomio cuadrático de componentes continuos + armónicos.',
        puerta='Si coordinador negativo, nivel=0 y no se construye diseño ni se llama a regresión.',seleccion_nivel='MAE global de validación; se informa también MAE en positivos detectados y positivos reales incluyendo FN.',
        fuentes_sha256={p:sha(ROOT/p) for p in [GEO_SOURCE,TERRAIN_SOURCE,REF_SOURCE]},registro_folds_sha256=hashlib.sha256(oof[['fila_excel','fold']].to_csv(index=False).encode()).hexdigest())
    write(OUT/'protocolo_comun.json',protocol)
    return d,m,w,split,fold,fold_context,data_audit,protocol

def fit_family(family,d,m,w,split,fold,fold_context,exchange=None):
    tr=split=='train';va=split=='validacion';yb=d[g.TARGET].to_numpy(float)>0;inputs=[m[b.CONTEXT],w,d[c.FOREST_FEATURES]]
    oof=np.zeros((tr.sum(),3));prob=np.zeros((len(d),3));models=[];programs=[]
    for j,name in enumerate(['contexto','antecedentes','individual']):
        for k in sorted(np.unique(fold)):
            a=fold!=k;z=fold==k;x=fold_context[int(k)] if j==0 else inputs[j].loc[tr].reset_index(drop=True)
            model=e.base_classifier(family,j,list(x),42+int(k))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(x.iloc[a],yb[tr][a])
            if any(issubclass(v.category,ConvergenceWarning) for v in caught):raise RuntimeError('No convergió '+family+' '+name)
            oof[z,j]=model.predict_proba(x.iloc[z])[:,1]
            print('OOF',family,name,int(k),flush=True)
        model=e.base_classifier(family,j,list(inputs[j]),42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(inputs[j].loc[tr],yb[tr])
        if any(issubclass(v.category,ConvergenceWarning) for v in caught):raise RuntimeError('No convergió ajuste completo '+family+' '+name)
        models.append(model);prob[:,j]=model.predict_proba(inputs[j])[:,1]
        if family=='GP':programs.append(dict(vista=name,expresion=model[-1].expression_,terminales=model[0].get_feature_names_out().tolist()))
    nets=h.VariationalStateNN().fit(oof,yb[tr]);candidates=[]
    for epoch,net in nets.items():
        p=net.predict_details(prob[va]).probabilidad_roya.to_numpy();quality=b.prior.probability_stats(yb[va],p)
        for row in b.prior.score_policies(yb[va],p):candidates.append(dict(epoch=epoch,**row,**quality))
    chosen=min(candidates,key=lambda r:(r['FP']+r['FN'],r['Brier'],-r['exactitud_balanceada']));net=nets[chosen['epoch']]
    selection=dict(familia=family,red=chosen,criterio='FP+FNvalidación; empateBrier/balanceada',programas=programs)
    g.csv(pd.DataFrame(candidates),OUT/('candidatos_'+family+'.csv'));write(OUT/('seleccion_'+family+'_antes_prueba.json'),selection)
    if exchange:exchange.classifier_ready(family,selection)
    obj=dict(models=models,bnn=net,selection=selection);g.save_model(obj,OUT/('clasificadores_'+family+'.joblib'))
    details=net.predict_details(prob);p=details.probabilidad_roya.to_numpy();gate=p>=chosen['umbral']
    rows=pd.concat([d[['fila_excel']].reset_index(drop=True),details],axis=1);rows['presencia_predicha']=gate;rows['conjunto']=split;rows['referencia_mayoria']=h.initial_decision(prob)
    for j,name in enumerate(['contexto','antecedentes','individual']):rows['p_'+name]=prob[:,j]
    g.csv(rows,OUT/('probabilidades_'+family+'.csv'))
    t=pd.DataFrame(oof,columns=['p_contexto','p_antecedentes','p_individual']);t['fila_excel']=d.loc[tr,'fila_excel'].to_numpy();t['fold']=fold;t['presencia_real']=yb[tr];g.csv(t,OUT/('OOF_'+family+'.csv'))
    return obj,rows

def quantity_metrics(y,p,gate):
    actual=y>0;tp=actual&gate
    return dict(MAE=float(mean_absolute_error(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),
        MAE_positivos_reales=float(mean_absolute_error(y[actual],p[actual])) if actual.any() else None,
        MAE_positivos_detectados=float(mean_absolute_error(y[tp],p[tp])) if tp.any() else None,
        n_positivos_detectados=int(tp.sum()),n_FN=int((actual&~gate).sum()),n_regresados=int(gate.sum()))

def run(exchange=None):
    threadpool_limits(2);d,m,w,split,fold,fold_context,data_audit,protocol=prepare();y=d[g.TARGET].to_numpy(float);yb=y>0
    tr=split=='train';va=split=='validacion';te=split=='prueba';families={};preds={}
    for family in FAMILIES:families[family],preds[family]=fit_family(family,d,m,w,split,fold,fold_context,exchange)
    qframe=e.quantity_frame(d,m,w);positive_train=tr&yb
    quantity=e.fit_regressions(qframe.loc[positive_train].reset_index(drop=True),y[positive_train]);g.save_model(quantity,OUT/'regresiones.joblib')
    g.csv(pd.DataFrame(quantity['metadata']),OUT/'ajustes_seis_regresiones.csv')
    valrows=[];selection={};values={}
    for family in FAMILIES:
        gate=preds[family].presencia_predicha.to_numpy(bool);values[family]={}
        for name in e.REGRESSIONS:
            # Evaluación primero en validación; negativos no se envían al modelo.
            q=e.predict_gated(quantity,name,qframe.loc[va].reset_index(drop=True),gate[va]);values[family][name]=np.zeros(len(d));values[family][name][va]=q
            valrows.append(dict(familia=family,regresion=name,conjunto='validacion',**quantity_metrics(y[va],q,gate[va])))
        selection[family]=min([v for v in valrows if v['familia']==family],key=lambda v:(v['MAE'],v['RMSE']))['regresion']
    preferred_class=min(FAMILIES,key=lambda f:(families[f]['selection']['red']['FP']+families[f]['selection']['red']['FN'],families[f]['selection']['red']['Brier']))
    preferred_quantity=min(valrows,key=lambda r:(r['MAE'],r['RMSE']))
    final_selection=dict(regresion_por_familia=selection,mejor_clasificacion=preferred_class,mejor_nivel=preferred_quantity,
        criterios='Clasificación:FP+FNvalidación. Magnitud:MAEglobal de validación. Prueba no selecciona.',
        PCA_continuas=quantity['design'].n_components_,varianza_PCA=quantity['design'].variance_,binarias_sin_ACP=quantity['design'].bins,
        minimo_positivo_train=quantity['min_positive'],cota='Tras la puerta positiva, predicción truncada al mínimo positivo observado en train; negativos=0.')
    write(OUT/'seleccion_integral_antes_prueba.json',final_selection);g.csv(pd.DataFrame(valrows),OUT/'regresiones_validacion.csv')
    if exchange:exchange.quantity_ready(final_selection)
    print('seleccion_integral_congelada',final_selection,flush=True)
    regrows=valrows.copy();classification=[];quality=[];baselines=[]
    for family in FAMILIES:
        frame=preds[family];gate=frame.presencia_predicha.to_numpy(bool);p=frame.probabilidad_roya.to_numpy()
        for tag,mask in [('validacion',va),('prueba',te)]:
            classification.append(dict(familia=family,conjunto=tag,**common.metrics(yb[mask],gate[mask])))
            quality.append(dict(familia=family,conjunto=tag,**b.prior.probability_stats(yb[mask],p[mask])))
        for name in e.REGRESSIONS:
            for tag,mask in [('train',tr),('prueba',te)]:
                q=e.predict_gated(quantity,name,qframe.loc[mask].reset_index(drop=True),gate[mask]);values[family][name][mask]=q
                if tag=='prueba':regrows.append(dict(familia=family,regresion=name,conjunto=tag,**quantity_metrics(y[mask],q,gate[mask])))
            frame['nivel_'+name]=values[family][name]
        frame['incidencia_predicha']=frame['nivel_'+selection[family]];frame['incidencia_observada']=y
        frame['P_error_final']=np.where(gate,1-p,p);frame['P_correcto_final']=1-frame.P_error_final
        frame['regresion_aplicada']=gate;frame['estado']=np.where(gate,'nivel_estimado','sin_roya_sin_regresion')
        g.csv(frame,OUT/('predicciones_'+family+'.csv'))
        median=float(np.median(y[positive_train]))
        for tag,mask in [('validacion',va),('prueba',te)]:
            for label,base in [('siempre_cero',np.zeros(mask.sum())),('mediana_positiva_misma_puerta',np.where(gate[mask],median,0))]:
                baselines.append(dict(familia=family,conjunto=tag,modelo=label,**quantity_metrics(y[mask],base,gate[mask])))
    g.csv(pd.DataFrame(classification),OUT/'matrices_y_metricas_clasificacion.csv');g.csv(pd.DataFrame(quality),OUT/'calidad_probabilidades.csv');g.csv(pd.DataFrame(regrows),OUT/'seis_regresiones.csv');g.csv(pd.DataFrame(baselines),OUT/'referencias_simples.csv')
    registry=dict(filas=d.fila_excel.astype(int).tolist(),source_sha256=protocol['fuentes_sha256'],seleccion=final_selection)
    write(OUT/'registro_modelo.json',registry)
    report=dict(auditoria=data_audit,protocolo=protocol,seleccion=final_selection,clasificacion=classification,probabilidades=quality,
        regresiones=regrows,referencias=baselines,ajustes_regresion=quantity['metadata'],
        limites=['Prueba histórica reutilizada: estudio exploratorio, no validación prospectiva.',
            'Muestreos17–24en cero y unidad de incidencia pendientes de verificar.',
            '200coordenadas tienen atributos topográficos contradictorios entre parcelas; se preservan por parcela, sin resolver su veracidad física.',
            'Fenología multietiqueta: cero indicadores activos no se interpreta automáticamente como dato perdido.',
            'Humedad real quincenal, no humedad simulada diaria; fechas aproximadas1/16.',
            'Sólo se predicen registros de la base congelada; no se permiten grillas ni escenarios externos en esta ruta.',
            'Polinomios sobre componentes continuos; indicadores y comunidades estáticas aditivos. Armónicos anuales1y2.',
            'OLS puede tener columnas redundantes por comunidades; se informa rango de matriz. Los coeficientes no prueban causalidad.',
            'La red topográfica nueva se ajusta porfold. Otras comunidades conservan covariables train comunes, sin enfermedad como entrada.',
            'La mejora de magnitud debe juzgarse junto a la matriz de clasificación y referencias simples.'])
    write(OUT/'resultados.json',report);make_report(report);plot(report);return report

def registered_data(request):
    registry=json.loads((OUT/'registro_modelo.json').read_text())
    for relative,digest in registry['source_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Cambió una fuente de la base; se requiere actualizar y reentrenar.')
    if 'fila_excel' not in request:raise ValueError('Esta ruta sólo acepta registros con fila_excel de la base.')
    ids=pd.to_numeric(request.fila_excel,errors='raise')
    if ids.isna().any() or not np.isfinite(ids).all() or (ids%1!=0).any():raise ValueError('fila_excel inválida.')
    if not ids.isin(registry['filas']).all():raise ValueError('Se rechazó un registro fuera de la base.')
    d=assemble().drop(columns=[g.TARGET]).set_index('fila_excel',drop=False).loc[ids.astype(int)].reset_index(drop=True)
    for col in request:
        if col in d and col!='fila_excel':
            a=request[col].reset_index(drop=True);z=d[col]
            if pd.api.types.is_numeric_dtype(z):same=np.allclose(pd.to_numeric(a,errors='coerce'),z,equal_nan=True,rtol=1e-8,atol=1e-8)
            else:same=a.fillna('').astype(str).equals(z.fillna('').astype(str))
            if not same:raise ValueError('Los valores de '+col+' no coinciden con el registro de la base.')
    return d,registry

def predict_registered(request,family='RF'):
    d,registry=registered_data(request);selection=registry['seleccion']
    if family=='mejor_clasificacion':family=selection['mejor_clasificacion']
    if family=='mejor_nivel':family=selection['mejor_nivel']['familia']
    model=joblib.load(OUT/('clasificadores_'+family+'.joblib'));ref=load_reference()
    m=pd.read_csv(OUT/'comunidades_por_fila.csv').set_index('fila_excel',drop=False).loc[d.fila_excel].reset_index(drop=True)
    w=h.features(d,ref['quantity']['bank'],ref['humidity_history']);inputs=[m[b.CONTEXT],w,d[c.FOREST_FEATURES]]
    prob=np.column_stack([v.predict_proba(x)[:,1] for v,x in zip(model['models'],inputs)]);result=model['bnn'].predict_details(prob)
    gate=result.probabilidad_roya.to_numpy()>=model['selection']['red']['umbral'];result['presencia_predicha']=gate;result['regresion_aplicada']=gate
    qframe=e.quantity_frame(d,m,w);quantity=joblib.load(OUT/'regresiones.joblib')
    for name in e.REGRESSIONS:result['nivel_'+name]=e.predict_gated(quantity,name,qframe,gate)
    chosen=selection['regresion_por_familia'][family];result['incidencia_predicha']=result['nivel_'+chosen];result['familia']=family
    result['fila_excel']=d.fila_excel.to_numpy();result['regresion_elegida']=chosen;result['estado']=np.where(gate,'nivel_estimado','sin_roya_sin_regresion')
    return result

def make_report(r):
    lines=['# Reentrenamiento integral: fenología binaria, topografía y seis regresiones','',
        'Se reentrenan las tres familias y sus coordinadores con el mismo protocolo. Fenología: seis indicadores binarios independientes; orientación circular, pendiente, coordenadas y altitud originales añadidas a la red topográfica.','',
        'La red topográfica tiene nodos parcela+coordenada por atributos contradictorios en200coordenadas. La partición mantiene juntas las coordenadas compartidas. OOF ajusta la topografía nueva dentro de cadafold.','',
        '## Puerta e inferencia','',
        'Si el coordinador indica ausencia, nivel=0 y no se llama a ninguna regresión. Si indica presencia, se aplican las seis regresiones sólo al registro existente. No hay grillas ni respuestas sintéticas. Los ajustes de cantidad usan sólo positivos reales de train.','',
        '## Matrices de clasificación','',pd.DataFrame(r['clasificacion']).to_markdown(index=False),'',
        '## Seis modelos de magnitud','',
        '1. Lineal múltiple por OLS.\n2. log(1+y)=lineal múltiple; inversiónexpm1.\n3. Polinomial2.\n4. Polinomial3.\n5. Lineal con seno/coseno anuales, armónicos1y2.\n6. Polinomial2 más los mismos armónicos.','',
        'ACP95% sólo en continuas; fenología y demás indicadores fuera deACP y expansiones, como efectos aditivos. Comunidades estáticas nominales; se excluyen dummies de periodo para contrastar armónicos. Polinomios de los componentes continuos. Todos son OLS, no Ridge.','',
        pd.DataFrame(r['regresiones']).to_markdown(index=False),'',
        'MAE global incluye negativos, falsas alarmas y positivos omitidos; en estos últimos permanececero, sin ejecutar regresión. También se informa MAE sobre verdaderos positivos detectados.','',
        '## Selección anterior a prueba','',json.dumps(r['seleccion'],ensure_ascii=False,indent=2),'',
        '## Referencias simples','',pd.DataFrame(r['referencias']).to_markdown(index=False),'',
        '## Ajustes y rango','',pd.DataFrame(r['ajustes_regresion']).to_markdown(index=False),'',
        '## Límites','']+['- '+v for v in r['limites']]
    (OUT/'INFORME.md').write_text('\n'.join(lines))

def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for tag in ['validacion','prueba']:
        rows=[v for v in r['clasificacion'] if v['conjunto']==tag];fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True);vmax=max(v['TN'] for v in rows)
        for ax,v in zip(axes,rows):
            a=np.array([[v['TN'],v['FP']],[v['FN'],v['TP']]]);ax.imshow(a,cmap='Blues',vmin=0,vmax=vmax);ax.set_title(LABELS[v['familia']]+' + BNN')
            ax.set_xticks([0,1],['Sin roya','Con roya']);ax.set_yticks([0,1],['Sin roya','Con roya']);ax.set_xlabel('Predicción');ax.set_ylabel('Estado real')
            for (i,j),n in np.ndenumerate(a):ax.text(j,i,f'{n:,}',ha='center',va='center',fontsize=20,color='white' if n>vmax/2 else '#17334b')
        fig.suptitle('Topografía original y fenología binaria · '+tag);fig.savefig(OUT/('matrices_'+tag+'.png'),dpi=180);plt.close(fig)

if __name__=='__main__':
    from reentrenamiento_integral_roya import run as canonical_run
    canonical_run()
