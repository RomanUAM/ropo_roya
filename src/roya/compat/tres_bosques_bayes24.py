"""Tres vistas de roya y posterior conjunto OOF, ventana estrictamente previa de24d.

La simulación disponible sólo mide aptitud térmica. No se fabrica humedad.
"""
import json,itertools
import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.sparse import csr_matrix,save_npz
from sklearn.metrics import balanced_accuracy_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
import dos_bosques_consenso_bayes as prior

c=prior.c;g=c.g;ROOT=c.ROOT;OUT=ROOT/'evaluacion_tres_bosques24'
CONTEXT=['comunidad_topografia','comunidad_ubicacion','comunidad_accidentes','comunidad_hidrica','comunidad_simulada','comunidad_urbanistica']
EXTRA={'accidentes':['Dist_falla_cartografiada_m','Dist_fractura_cartografiada_m','Posicion_relativa_MDE_315m_m','Dist_relieve_alto_relativo_m'],
       'hidrica':['Dist_cauce_cartografiado_m']}
REGRESSIONS=['lineal_multiple','log1p_lineal']

class JointBeta3(prior.JointBetaConsensus):
    def fit(self,p,y):
        super().fit(p,y)
        p=np.asarray(p,float);y=np.asarray(y,float);changed=False
        # Dos valores repetidos pueden dejar todos los cuantiles en extremos.
        # En tal caso conservar una separación real, en lugar de perder el eje.
        for i,col in enumerate(p.T):
            if not len(self.cuts_[i]) and col.min()<col.max():
                self.cuts_[i]=np.array([(col.min()+col.max())/2]);changed=True
        if changed:
            shape=tuple(len(v)+1 for v in self.cuts_);self.n_=np.zeros(shape);self.k_=np.zeros(shape)
            ix=self.indices(p);np.add.at(self.n_,ix,1);np.add.at(self.k_,ix,y)
            self.a_=self.a0_+self.k_;self.b_=self.b0_+self.n_-self.k_
        return self
    def details(self,p):
        ix=self.indices(p);a=self.a_[ix];b=self.b_[ix]
        return pd.DataFrame(dict(celda_contexto=ix[0],celda_24dias=ix[1],celda_individual=ix[2],
            observaciones_celda=self.n_[ix].astype(int),probabilidad_roya=a/(a+b),
            limite_inferior95=beta.ppf(.025,a,b),limite_superior95=beta.ppf(.975,a,b)))

def window24(d,bank):
    """No incluye hoy; faltantes previos al año son NaN y no ausencia."""
    d=d.reset_index(drop=True)
    if 'Fecha_muestreo_exacta' in d and d.Fecha_muestreo_exacta.notna().any():
        raise ValueError('La referencia usa quincenas; requiere reconstrucción para fechas exactas.')
    dates=bank['dates'];ids=g.site_ids(d).map(bank['xy'])
    if ids.isna().any():raise ValueError('Ubicación sin simulación original.')
    ids=ids.to_numpy(int);result=pd.DataFrame(index=d.index)
    for process in ['germinacion','apresorio']:
        vals=np.full((len(d),24),np.nan)
        for period,ix in d.groupby(g.periods(d)).groups.items():
            date=pd.Timestamp(period[:7]+'-'+('01' if period[-1]=='1' else '16'))
            if date not in dates:raise ValueError('Fecha fuera del año simulado; no se usa módulo anual.')
            day=dates.get_loc(date);ix=np.asarray(ix)
            for lag in range(1,25):
                if day>=lag:vals[ix,lag-1]=(bank['arrays'][process+'_respuesta_realizacion_1'][day-lag,ids[ix]]>=.9).astype(float)
        for lag in range(1,25):result[f'{process}_termica_dmenos{lag}']=vals[:,lag-1]
        # Suma parcial explícita; nunca se trata una ventana incompleta como24d.
        result[process+'_dias_observados']=np.isfinite(vals).sum(axis=1)
        result[process+'_dias_favorables_observados']=np.nansum(vals,axis=1)
    result['ventana24_incompleta']=(result.germinacion_dias_observados<24).astype(int)
    return result

def extra_graphs(d,split,m):
    u=d.assign(_sitio=g.site_ids(d)).drop_duplicates('_sitio').reset_index(drop=True)
    train_sites=set(g.site_ids(d[split=='train']));fit=u._sitio.isin(train_sites).to_numpy();models={};reports=[]
    for name,features in EXTRA.items():
        model=g.SimilarityLayer();w=model.fit(u.loc[fit,features]);models[name]=dict(model=model,train_ids=u.loc[fit,'_sitio'].tolist())
        save_npz(OUT/('arcos_'+name+'.npz'),csr_matrix(w));labels=g.assign(model.weights(u[features]),model.labels,name+'_')
        labels[fit]=[name+'_'+str(k) for k in model.labels]
        m['comunidad_'+name]=g.site_ids(d).map(dict(zip(u._sitio,labels)))
        reports.append(dict(capa=name,variables=features,**model.report))
    return models,reports

def extra_transform(d,m,models):
    u=d.assign(_sitio=g.site_ids(d)).drop_duplicates('_sitio').reset_index(drop=True)
    for name,item in models.items():
        model=item['model'];labels=g.assign(model.weights(u[EXTRA[name]]),model.labels,name+'_')
        ref=dict(zip(item['train_ids'],model.labels))
        for i,s in enumerate(u._sitio):
            if s in ref:labels[i]=name+'_'+str(ref[s])
        m['comunidad_'+name]=g.site_ids(d).map(dict(zip(u._sitio,labels)))
    return m

def outcomes(p,decision):
    """Orden: estado real y decisión predicha; decisión conocida =>2ceros."""
    a=np.asarray(decision,bool);p=np.asarray(p,float)
    return pd.DataFrame({'P_real_neg_pred_neg':np.where(~a,1-p,0),
        'P_real_pos_pred_neg':np.where(~a,p,0),'P_real_neg_pred_pos':np.where(a,1-p,0),
        'P_real_pos_pred_pos':np.where(a,p,0)})

def votes_table(oof,y,thresholds,model):
    votes=oof>=thresholds;rows=[]
    for pattern in itertools.product([False,True],repeat=3):
        mask=(votes==pattern).all(axis=1);n=int(mask.sum());k=int(y[mask].sum());a=model.a0_+k;b=model.b0_+n-k
        rows.append(dict(contexto=pattern[0],previos24=pattern[1],individual=pattern[2],n_OOF=n,positivos_OOF=k,
                         probabilidad_roya=a/(a+b),inferior95=float(beta.ppf(.025,a,b)),superior95=float(beta.ppf(.975,a,b))))
    return pd.DataFrame(rows)

def run(exchange=None):
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    d=g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(g.OUT/'comunidades_por_observacion.csv');assert np.array_equal(d.fila_excel,m.fila_excel)
    split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';y=d[g.TARGET].to_numpy(float);yb=y>0
    old=joblib.load(c.OUT/'modelo.joblib');quantity=old['quantity'];bank=quantity['bank']
    extra,reports=extra_graphs(d,split,m);g.csv(pd.DataFrame(reports),OUT/'redes_adicionales.csv')
    win=window24(d,bank);g.csv(pd.concat([d[['fila_excel']],win],axis=1),OUT/'ventana24.csv')
    previous=pd.read_csv(c.OUT/'objetivos_OOF.csv');assert np.array_equal(previous.fila_excel,d.loc[tr,'fila_excel'])
    fold=previous.fold.to_numpy();sites=g.site_ids(d[tr]).to_numpy();oof=np.zeros((tr.sum(),3));oof[:,2]=previous.probabilidad_OOF
    inputs=[m[CONTEXT],win,d[c.FOREST_FEATURES]];models=[];thresholds=[];prob=np.zeros((len(d),3))
    for view in [0,1]:
        factory=prior.global_forest if view==0 else c.forest
        for k in sorted(np.unique(fold)):
            a=fold!=k;b=fold==k;assert not set(sites[a])&set(sites[b])
            rf=factory(42+int(k)).fit(inputs[view][tr].iloc[a],yb[tr][a]);oof[b,view]=rf.predict_proba(inputs[view][tr].iloc[b])[:,1]
            print('OOF',view,int(k),flush=True)
        model=factory().fit(inputs[view][tr],yb[tr]);models.append(model);prob[:,view]=model.predict_proba(inputs[view])[:,1]
    models.append(old['forest']);prob[:,2]=models[2].predict_proba(inputs[2])[:,1]
    for j in range(3):thresholds.append(float(max(np.arange(.05,.951,.025),key=lambda t:balanced_accuracy_score(yb[tr],oof[:,j]>=t))))
    candidates={};resolutions=[]
    # Menos resoluciones que la rama2D: evitar multiplicar celdas casi vacías.
    for bins in [3,5]:
        key='beta3_'+str(bins);model=JointBeta3(bins,20).fit(oof,yb[tr]);candidates[key]=model
        resolutions.append(dict(modelo=key,celdas=int(model.n_.size),**prior.probability_stats(yb[va],model.predict(prob[va]))))
    selected=min(resolutions,key=lambda r:r['Brier'])['modelo'];bayes=candidates[selected];p=bayes.predict(prob)
    policies=prior.score_policies(yb[va],p[va]);policy=max(policies,key=lambda r:(r['exactitud_balanceada'],-r['FP'],-r['FN']))
    controls=[dict(nombre=['contexto','previos24','individual'][j],**g.stats(yb[va],prob[va,j]>=thresholds[j])) for j in range(3)]
    reference=max(controls,key=lambda r:r['exactitud_balanceada']);safe=prior.recommend(policies,reference)
    selection=dict(modelo=selected,politica_balanceada=policy,referencia=reference,politica_dominante=safe,
                   regla='Elegir resolución por Brier validación; umbral por exactitud balanceada validación. Sin optimizar prueba.')
    g.csv(pd.DataFrame(resolutions),OUT/'resoluciones_validacion.csv');g.csv(pd.DataFrame(policies),OUT/'politicas_validacion.csv')
    (OUT/'seleccion_antes_prueba.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    if exchange:exchange.validation_ready(selection)
    print('seleccion_congelada',selection,flush=True)
    decisions=prob>=np.array(thresholds);decision=p>=policy['umbral'];comparison=[];quality=[]
    for tag,mask in [('validacion',va),('prueba',te)]:
        for j,name in enumerate(['contexto','previos24','individual']):
            comparison.append(dict(conjunto=tag,modelo=name,**g.stats(yb[mask],decisions[mask,j])))
            quality.append(dict(conjunto=tag,modelo=name,**prior.probability_stats(yb[mask],prob[mask,j])))
        comparison.append(dict(conjunto=tag,modelo='consenso3',**g.stats(yb[mask],decision[mask])))
        quality.append(dict(conjunto=tag,modelo='consenso3',**prior.probability_stats(yb[mask],p[mask])))
    # Coeficientes y ACP de cantidad ya entrenados sólo en positivos del mismo train.
    z=quantity['reg_representation'].transform(g.tabular(d,m,'combinadas'));reg_rows=[];quantities={}
    for name in REGRESSIONS:
        q=np.where(decision,g.reg_predict(quantity['regressors'][name],name,z,d),0);quantities[name]=q
        for tag,mask in [('validacion',va),('prueba',te)]:
            reg_rows.append(dict(modelo=name,conjunto=tag,MAE=float(mean_absolute_error(y[mask],q[mask])),RMSE=float(np.sqrt(mean_squared_error(y[mask],q[mask])))))
    chosen=min([r for r in reg_rows if r['conjunto']=='validacion'],key=lambda r:r['MAE'])['modelo']
    bundle=dict(models=models,thresholds=thresholds,bayes=bayes,selection=selection,extra_graphs=extra,quantity=quantity,
                selected_regression=chosen,window_features=win.columns.tolist(),humidity_available=False)
    rows=pd.concat([d[['fila_excel',g.TARGET]].reset_index(drop=True),bayes.details(prob),outcomes(p,decision)],axis=1)
    rows['conjunto']=split;rows['presencia_predicha']=decision
    for j,name in enumerate(['contexto','previos24','individual']):rows['p_'+name]=prob[:,j];rows['voto_'+name]=decisions[:,j]
    for name,q in quantities.items():rows['incidencia_'+name]=q
    rows['incidencia_predicha']=quantities[chosen];rows['humedad_simulada_disponible']=False
    g.csv(rows,OUT/'predicciones.csv');g.csv(pd.DataFrame(comparison),OUT/'comparacion.csv');g.csv(pd.DataFrame(quality),OUT/'calidad_probabilidades.csv');g.csv(pd.DataFrame(reg_rows),OUT/'dos_regresiones.csv')
    o=pd.DataFrame(oof,columns=['p_contexto','p_previos24','p_individual']);o['fila_excel']=d.loc[tr,'fila_excel'].to_numpy();o['fold']=fold;o['presencia_real']=yb[tr];g.csv(o,OUT/'entradas_OOF.csv')
    votes=votes_table(oof,yb[tr],thresholds,bayes);g.csv(votes,OUT/'ocho_combinaciones.csv')
    g.save_model(bundle,OUT/'modelo.joblib')
    report=dict(diseno='Mismas ubicaciones50/20/30 y cinco folds OOF por coordenadas; gráficos ajustados con covariables train (transductivo dentro de OOF), sin usar etiquetas de enfermedad.',
        bosques=dict(arboles=180,profundidad_maxima=12,minimo_hoja=15,peso_clase='balanced_subsample',semilla=42,
            normalizacion='Escalado antes de Mahalanobis; comunidades one-hot. Bosques numéricos con imputación mediana, sin ACP ni escalado necesario para cortes.',
            regresiones='ACP95 y escalado numérico previos, categorías nominales; coeficientes positivos existentes.'),
        entrada_contexto=CONTEXT,entrada24=win.columns.tolist(),entrada_individual=c.FOREST_FEATURES,
        humedad_simulada_disponible=False,estado='experimental_parcial_solo_aptitud_termica',
        umbrales_bosques=thresholds,seleccion=selection,resoluciones=resolutions,comparacion=comparison,calidad_probabilidades=quality,regresiones=reg_rows,regresion_elegida=chosen,
        ocho_combinaciones=votes.to_dict('records'),redes=reports,
        limites=['No existe humedad simulada: el bosque24 evalúa aptitud térmica con umbral heredado0.9; no infección validada.',
        'Sólo se predice roya observada; no presencia general de hongos.',
        'Fecha aproximada día1/16 por quincena. Ventana24 excluye hoy y conserva faltantes; sin módulo ni circularidad anual.',
        'Muestreos17–24 son cero; falta verificar si son ausencias reales. Prueba histórica reutilizada: resultados exploratorios.',
        'Los intervalos Beta condicionales no contemplan toda la correlación entre observaciones del mismo sitio.',
        'Comunidades simuladas dependen del periodo; nuevas fechas y sitios requieren reconstrucción y validación.',
        'Coordenadas y altitud tienen incertidumbres de referencia; aguas cartografiadas no son mojado foliar. Grafo hídrico de similitud de distancia a cauce, no conectividad física de ríos.',
        'Dos regresiones existentes mantienen sus coeficientes entrenados en positivos; se reevalúan con la nueva puerta. Las otras familias no se eliminan.'])
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));write_report(report)
    return report

def predict(bundle,d):
    d=g.enrich(bundle['quantity'],d).reset_index(drop=True);q=bundle['quantity']
    m=g.graph_transform(d,q['bank'],q['blocks']);m=extra_transform(d,m,bundle['extra_graphs']);w=window24(d,q['bank'])
    prob=np.column_stack([model.predict_proba(x)[:,1] for model,x in zip(bundle['models'],[m[CONTEXT],w,d[c.FOREST_FEATURES]])])
    details=bundle['bayes'].details(prob);p=details.probabilidad_roya.to_numpy();a=p>=bundle['selection']['politica_balanceada']['umbral']
    out=pd.concat([details,outcomes(p,a)],axis=1);out['presencia_predicha']=a;out['ventana24_incompleta']=w.ventana24_incompleta
    for j,name in enumerate(['contexto','previos24','individual']):out['p_'+name]=prob[:,j];out['voto_'+name]=prob[:,j]>=bundle['thresholds'][j]
    z=q['reg_representation'].transform(g.tabular(d,m,'combinadas'))
    for name in REGRESSIONS:out['incidencia_'+name]=np.where(a,g.reg_predict(q['regressors'][name],name,z,d),0)
    out['incidencia_predicha']=out['incidencia_'+bundle['selected_regression']]
    out['humedad_simulada_disponible']=False;out['estado']='experimental_parcial_solo_aptitud_termica'
    return out

def write_report(r):
    lines=['# Tres bosques y consenso bayesiano de roya','',
    'Estado: **experimental parcial**. Falta humedad simulada; no se inventó ni se sustituyó por humedad real.','',
    '1. Contexto: comunidades topográfica, ubicación (coordenadas y altitud), accidentes, hídrica, aptitud simulada actual y urbanística.\n2. Antecedentes:48 indicadores térmicos diarios (24germinación y24apresorio), cobertura y conteos observados.\n3. Individual: clima real y fenología. Los tres aprenden presencia/ausencia de roya.','',r['diseno'],'',
    'El consenso aprende P(roya | pContexto,p24,pIndividual) con predicciones OOF. Usa celdas conjuntas3D Beta-Bernoulli y previa empírica de20observaciones, sin multiplicar probabilidades independientes. Se comparan3/5cuantiles por eje; resolución por Brier de validación y corte por exactitud balanceada.','',
    'Los tres bosques tienen180árboles, profundidad máxima12 y mínimo15observaciones por hoja, con pesos de clase balanceados. Las distancias Mahalanobis usan escalado y covarianza regularizada; las comunidades se codifican nominalmente. Los bosques numéricos usan imputación sin ACP; las dos regresiones conservan ACP95 y escalado de su ajuste anterior. Los grafos son similitud entre sitios; el hídrico usa distancia al cauce cartografiado, sin inferir conexión física entre ríos.','',
    'Hay ocho patrones de voto; no se comprimen en cuatro. Las cuatro probabilidades diagnósticas usan orden **real, predicho**: negativo-negativo(TN), positivo-negativo(FN), negativo-positivo(FP), positivo-positivo(TP). Como la decisión ya es conocida, dos estados incompatibles tienen probabilidad cero y los otros suman uno. Se estiman riesgos, no diagnósticos verificados.','',
    '## Comparación','',pd.DataFrame(r['comparacion']).to_markdown(index=False),'',
    '## Calidad probabilística','',pd.DataFrame(r['calidad_probabilidades']).to_markdown(index=False),'',
    '## Dos regresiones','',pd.DataFrame(r['regresiones']).to_markdown(index=False),'','Elegida por validación: '+r['regresion_elegida']+'.','',
    '## Aprendizaje y mejoras propuestas','',
    '- Incorporar humedad simulada diaria y duración del mojado foliar antes de afirmar condiciones conjuntas favorables. Validar el umbral biológico.\n- Separar y verificar los ceros17–24 antes de interpretar falsas alarmas.\n- Validar en fecha/localidad nueva; esta prueba ya fue consultada muchas veces.\n- Evaluar ablaciones de cada bosque y de cada grafo con los mismos datos, sin añadir capas automáticamente.\n- Elegir el corte según costo de FN y FP. Mantener zona de revisión para celdas escasas y desacuerdos.\n- Reconstruir comunidades dentro de cada fold para medir cuánto influye el uso transductivo de covariables train.\n- Comparar regresiones contra mediana positiva y reportar error separado entre positivos detectados y positivos omitidos.','',
    '## Límites','']+['- '+v for v in r['limites']]
    (OUT/'INFORME.md').write_text('\n'.join(lines))

if __name__=='__main__':
    from tres_bosques_bayes24 import run as canonical_run
    canonical_run()
