"""Cuarta arquitectura heterogénea y séptimo estimador de cantidad.

Reutiliza modelos y OOF congelados; no vuelve a consultar prueba para elegir.
"""
import json,warnings
import joblib,numpy as np,pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
import reentrenamiento_integral_roya as old

e=old.e;h=old.h;g=old.g;b=old.b;ROOT=old.ROOT
OUT=ROOT/'evaluacion_cuatro_arquitecturas'
FAMILIES=['RF','LR','GP','MIXTA']
METHODS=e.REGRESSIONS+['red_neuronal']
VIEWS=['contexto','antecedentes','individual']

def data():
    d=old.assemble();m=pd.read_csv(old.OUT/'comunidades_por_fila.csv')
    assert np.array_equal(d.fila_excel,m.fila_excel)
    ref=old.load_reference();w=h.features(d,ref['quantity']['bank'],ref['humidity_history'])
    return d,m,w,ref

def fit_mixed(d,split):
    tr=split=='train';va=split=='validacion';y=d[g.TARGET].to_numpy()>0
    selected=[];candidate=[];inputs=[];oof=[]
    folds=None
    for family in old.FAMILIES:
        p=pd.read_csv(old.OUT/('probabilidades_'+family+'.csv'))
        o=pd.read_csv(old.OUT/('OOF_'+family+'.csv'))
        assert np.array_equal(p.fila_excel,d.fila_excel) and np.array_equal(o.fila_excel,d.loc[tr,'fila_excel'])
        if folds is None:folds=o.fold.to_numpy()
        assert np.array_equal(folds,o.fold)
        rows=[]
        for j,view in enumerate(VIEWS):
            row=dict(familia=family,vista=view,indice=j,**b.prior.probability_stats(y[va],p.loc[va,'p_'+view]))
            rows.append(row);candidate.append(row)
        chosen=min(rows,key=lambda r:(r['Brier'],r['log_loss'],r['indice']))
        selected.append(chosen);inputs.append(p['p_'+chosen['vista']].to_numpy());oof.append(o['p_'+chosen['vista']].to_numpy())
    old.write(OUT/'seleccion_entradas_antes_prueba.json',dict(criterio='Menor Brier de validación por familia; empate logloss. Una de RF, LR binomial y GP.',entradas=selected))
    g.csv(pd.DataFrame(candidate),OUT/'calidad_nueve_probabilidades_validacion.csv')
    prob=np.column_stack(inputs);oof=np.column_stack(oof)
    nets=h.VariationalStateNN().fit(oof,y[tr]);candidates=[]
    for epoch,net in nets.items():
        pv=net.predict_details(prob[va]).probabilidad_roya.to_numpy();quality=b.prior.probability_stats(y[va],pv)
        candidates.extend(dict(epoch=epoch,**row,**quality) for row in b.prior.score_policies(y[va],pv))
    chosen=min(candidates,key=lambda r:(r['FP']+r['FN'],r['Brier'],-r['exactitud_balanceada']))
    g.csv(pd.DataFrame(candidates),OUT/'candidatos_MIXTA.csv')
    obj=dict(entradas=selected,bnn=nets[chosen['epoch']],selection=chosen)
    old.write(OUT/'seleccion_MIXTA_antes_prueba.json',chosen);g.save_model(obj,OUT/'clasificadores_MIXTA.joblib')
    result=obj['bnn'].predict_details(prob);result['fila_excel']=d.fila_excel;result['conjunto']=split
    result['presencia_predicha']=result.probabilidad_roya>=chosen['umbral']
    for j,family in enumerate(old.FAMILIES):result['p_'+family]=prob[:,j]
    g.csv(result,OUT/'probabilidades_MIXTA.csv')
    out=pd.DataFrame(oof,columns=['p_'+f for f in old.FAMILIES]);out['fila_excel']=d.loc[tr,'fila_excel'].to_numpy();out['fold']=folds
    g.csv(out,OUT/'OOF_MIXTA.csv')
    return result

def fit_neural(frame,y,split,design):
    tr=(split=='train')&(y>0);va=(split=='validacion')&(y>0)
    scaler=StandardScaler().fit(design.transform('lineal_multiple',frame.loc[tr].reset_index(drop=True)))
    xt=scaler.transform(design.transform('lineal_multiple',frame.loc[tr].reset_index(drop=True)))
    xv=scaler.transform(design.transform('lineal_multiple',frame.loc[va].reset_index(drop=True)))
    candidates=[];best=None;bestkey=None
    # External grouped validation; no random internal early-stopping split.
    for layers in [(36,18),(36,18,9)]:
        net=MLPRegressor(hidden_layer_sizes=layers,activation='relu',solver='adam',alpha=.01,batch_size=256,
            learning_rate_init=.001,max_iter=1,warm_start=True,shuffle=True,random_state=42,early_stopping=False)
        for epoch in range(1,161):
            net.partial_fit(xt,np.log1p(y[tr]))
            if epoch in [40,80,160]:
                pred=np.maximum(1,np.expm1(np.clip(net.predict(xv),-30,30)))
                row=dict(capas=list(layers),epoch=epoch,MAE_positivos_validacion=float(mean_absolute_error(y[va],pred)),n_train=int(tr.sum()))
                candidates.append(row);key=(row['MAE_positivos_validacion'],sum(layers),epoch)
                if bestkey is None or key<bestkey:
                    import copy
                    best=copy.deepcopy(net);bestkey=key;selected=row
        print('red_cantidad',layers,candidates[-1],flush=True)
    g.csv(pd.DataFrame(candidates),OUT/'candidatos_red_cantidad.csv')
    obj=dict(model=best,scaler=scaler,seleccion=selected,min_positive=float(y[tr].min()))
    g.save_model(obj,OUT/'red_cantidad.joblib');old.write(OUT/'seleccion_red_cantidad_antes_prueba.json',selected)
    return obj

def neural_gated(net,design,frame,gate):
    gate=np.asarray(gate,bool);out=np.zeros(len(frame))
    if gate.any():
        x=net['scaler'].transform(design.transform('lineal_multiple',frame.loc[gate].reset_index(drop=True)))
        out[gate]=np.maximum(net['min_positive'],np.expm1(np.clip(net['model'].predict(x),-30,30)))
    return out

def predict_registered(request,family='RF',regression=None):
    """Cuatro rutas seleccionables, siete cantidades; nunca infiere fuera de la base."""
    d,registry=old.registered_data(request)
    selected=json.loads((OUT/'seleccion_antes_prueba.json').read_text())
    if family=='mejor_clasificacion':family=selected['mejor_clasificacion']
    if family=='mejor_nivel':family=selected['mejor_nivel']['familia']
    if family not in FAMILIES:raise ValueError('Arquitectura desconocida')
    m=pd.read_csv(old.OUT/'comunidades_por_fila.csv').set_index('fila_excel',drop=False).loc[d.fila_excel].reset_index(drop=True)
    ref=old.load_reference();w=h.features(d,ref['quantity']['bank'],ref['humidity_history']);views=[m[b.CONTEXT],w,d[old.c.FOREST_FEATURES]]
    if family=='MIXTA':
        model=joblib.load(OUT/'clasificadores_MIXTA.joblib');cols=[]
        for item in model['entradas']:
            base=joblib.load(old.OUT/('clasificadores_'+item['familia']+'.joblib'))
            j=item['indice'];cols.append(base['models'][j].predict_proba(views[j])[:,1])
        prob=np.column_stack(cols);threshold=model['selection']['umbral']
    else:
        model=joblib.load(old.OUT/('clasificadores_'+family+'.joblib'))
        prob=np.column_stack([z.predict_proba(x)[:,1] for z,x in zip(model['models'],views)]);threshold=model['selection']['red']['umbral']
    result=model['bnn'].predict_details(prob);gate=result.probabilidad_roya.to_numpy()>=threshold
    quantity=joblib.load(old.OUT/'regresiones.joblib');net=joblib.load(OUT/'red_cantidad.joblib');frame=e.quantity_frame(d,m,w)
    for name in e.REGRESSIONS:result['nivel_'+name]=e.predict_gated(quantity,name,frame,gate)
    result['nivel_red_neuronal']=neural_gated(net,quantity['design'],frame,gate)
    method=regression or selected['regresion_por_familia'][family]
    if method not in METHODS:raise ValueError('Regresión desconocida')
    result['incidencia_predicha']=result['nivel_'+method];result['regresion_elegida']=method
    result['familia']=family;result['fila_excel']=d.fila_excel.to_numpy();result['presencia_predicha']=gate;result['regresion_aplicada']=gate
    result['estado']=np.where(gate,'nivel_estimado','sin_roya_sin_regresion')
    return result

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    d,m,w,ref=data();split=m.conjunto.to_numpy();y=d[g.TARGET].to_numpy(float);yb=y>0
    va=split=='validacion';te=split=='prueba';frame=e.quantity_frame(d,m,w);quantity=joblib.load(old.OUT/'regresiones.joblib')
    preds={f:pd.read_csv(old.OUT/('probabilidades_'+f+'.csv')) for f in old.FAMILIES}
    preds['MIXTA']=fit_mixed(d,split);net=fit_neural(frame,y,split,quantity['design'])
    vals=[];gates={f:preds[f].presencia_predicha.to_numpy(bool) for f in FAMILIES}
    for f in FAMILIES:
        for method in METHODS:
            q=neural_gated(net,quantity['design'],frame.loc[va].reset_index(drop=True),gates[f][va]) if method=='red_neuronal' else e.predict_gated(quantity,method,frame.loc[va].reset_index(drop=True),gates[f][va])
            vals.append(dict(familia=f,regresion=method,conjunto='validacion',**old.quantity_metrics(y[va],q,gates[f][va])))
    cl=[];quality=[]
    for f in FAMILIES:
        cl.append(dict(familia=f,conjunto='validacion',**old.common.metrics(yb[va],gates[f][va])))
        quality.append(dict(familia=f,conjunto='validacion',**b.prior.probability_stats(yb[va],preds[f].loc[va,'probabilidad_roya'])))
    bestclass=min(cl,key=lambda r:(r['FP']+r['FN'],next(z['Brier'] for z in quality if z['familia']==r['familia'])))['familia']
    selection=dict(regresion_por_familia={f:min([r for r in vals if r['familia']==f],key=lambda r:(r['MAE'],r['RMSE']))['regresion'] for f in FAMILIES},mejor_clasificacion=bestclass,mejor_nivel=min(vals,key=lambda r:(r['MAE'],r['RMSE'])))
    old.write(OUT/'seleccion_antes_prueba.json',selection);g.csv(pd.DataFrame(vals),OUT/'validacion_regresiones.csv')
    print('seleccion_congelada',selection,flush=True)
    metrics=vals.copy();baselines=[]
    for f in FAMILIES:
        r=preds[f];gate=gates[f];r['incidencia_observada']=y
        cl.append(dict(familia=f,conjunto='prueba',**old.common.metrics(yb[te],gate[te])))
        quality.append(dict(familia=f,conjunto='prueba',**b.prior.probability_stats(yb[te],r.loc[te,'probabilidad_roya'])))
        for method in METHODS:
            # Existing six fits are exactly shared with the preceding comparison.
            q=neural_gated(net,quantity['design'],frame,gate) if method=='red_neuronal' else e.predict_gated(quantity,method,frame,gate)
            r['nivel_'+method]=q
            metrics.append(dict(familia=f,regresion=method,conjunto='prueba',**old.quantity_metrics(y[te],q[te],gate[te])))
        r['incidencia_predicha']=r['nivel_'+selection['regresion_por_familia'][f]];r['regresion_aplicada']=gate
        g.csv(r,OUT/('predicciones_'+f+'.csv'))
        for tag,mask in [('validacion',va),('prueba',te)]:
            for name,p in [('siempre_cero',np.zeros(mask.sum())),('mediana_misma_puerta',np.where(gate[mask],np.median(y[(split=='train')&yb]),0))]:
                baselines.append(dict(familia=f,conjunto=tag,modelo=name,**old.quantity_metrics(y[mask],p,gate[mask])))
    g.csv(pd.DataFrame(metrics),OUT/'siete_regresiones.csv');g.csv(pd.DataFrame(cl),OUT/'clasificacion.csv');g.csv(pd.DataFrame(quality),OUT/'calidad_probabilidades.csv');g.csv(pd.DataFrame(baselines),OUT/'referencias.csv')
    report=dict(seleccion=selection,clasificacion=cl,calidad=quality,regresiones=metrics,referencias=baselines,red_cantidad=net['seleccion'],
        protocolo=dict(particion='Mismo50/20/30 por coordenada, mismos5folds OOF; train28728/val11520/test17352.',
        mixto='Una vista por familia con menor Brier de validación entre contexto, antecedentes e individual; tres probabilidades OOF de train alimentan BNN3-8-4.',
        seleccion='Vistas por Brier; coordinador/checkpoint/umbral por FP+FN validación, empateBrier; cantidad por MAEglobal validación. Prueba no selecciona.',
        red_cantidad='MLP log1p(y): candidatos36-18 y36-18-9, checkpoints40/80/160; sólo11900positivos train; ACP13continuas ya ajustada train + indicadores/categorías. Selección por MAE de positivos reales de validación. Adam,alpha0.01,batch256,seed42; sin early stopping aleatorio.',
        cantidad='Seis ajustesOLS anteriores preservados y compartidos entre cuatro puertas; una nueva red compartida. No ejecutar cantidades para negativos; sólofila_excel registrada.',
        limitaciones='Prueba histórica reutilizada, exploratoria. Selección de vista y coordinador comparten validación; falta validación anidada y cohorte independiente. Comunidades no topográficas conservan representación transductiva train.'))
    old.write(OUT/'resultados.json',report)
    from correlaciones_roya import run as corr
    corr(d,w,ref,split)
    return report

if __name__=='__main__':
    from cuatro_arquitecturas_roya import run as canonical_run
    canonical_run()
