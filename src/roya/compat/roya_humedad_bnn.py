"""Tres bosques con HR real y red bayesiana variacional de cuatro estados.

75/80% son hipótesis del usuario, no umbrales universales de infección.
20–24d son rezagos ambientales; no observación de colonización interna.
"""
import copy,json
import joblib,numpy as np,pandas as pd
from scipy.special import expit,logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
import tres_bosques_bayes24 as b

c=b.c;g=b.g;ROOT=b.ROOT;OUT=ROOT/'evaluacion_humedad_bnn'
CODES=np.array([-1,0,1,2]) # FN,TN,TP,FP respecto a mayoría inicial.

def keys(d):return d.Municipio.astype(str)+'|'+d.Parcela.astype(str)+'|'+g.site_ids(d)

def reference_dates(d):
    return pd.to_datetime(g.periods(d).map(lambda p:p[:7]+'-'+('01' if p[-1]=='1' else '16')))

def humidity_history(d):
    rh=pd.to_numeric(d.Humedad_relativa,errors='raise')
    if not np.isfinite(rh).all() or not rh.between(0,100).all():raise ValueError('HR real debe estar entre0y100%.')
    h=pd.DataFrame({'clave':keys(d),'fecha':reference_dates(d),'HR':rh})
    if h.groupby(['clave','fecha']).HR.nunique().max()>1:raise ValueError('HR contradictoria para mismo sitio/parcela/fecha.')
    return h.drop_duplicates(['clave','fecha']).reset_index(drop=True)

def features(d,bank,history):
    d=d.reset_index(drop=True);w=b.window24(d,bank);rh=pd.to_numeric(d.Humedad_relativa,errors='raise')
    if not np.isfinite(rh).all() or not rh.between(0,100).all():raise ValueError('HR real debe expresarse en%, entre0y100.')
    w['HR_real_actual_pct']=rh
    for cut in [75,80]:w[f'HR_real_actual_ge{cut}']=(rh>=cut).astype(float)
    h=history.copy();h['fecha']=pd.to_datetime(h.fecha);dates=reference_dates(d);ks=keys(d)
    for name in ['HR_previa24_media','HR_previa24_min','HR_previa24_max','HR_previa24_edad_ultima_dias',
                 'HR_previa24_n_mediciones','HR_previa24_n_mediciones_ge75','HR_previa24_n_mediciones_ge80']:
        w[name]=np.nan
    for date,ix in d.groupby(dates).groups.items():
        prev=h[(h.fecha>=date-pd.Timedelta(days=24))&(h.fecha<date)]
        agg=prev.assign(ge75=(prev.HR>=75).astype(int),ge80=(prev.HR>=80).astype(int)).groupby('clave').agg(
            HR_previa24_media=('HR','mean'),HR_previa24_min=('HR','min'),HR_previa24_max=('HR','max'),
            HR_previa24_n_mediciones=('HR','size'),HR_previa24_n_mediciones_ge75=('ge75','sum'),
            HR_previa24_n_mediciones_ge80=('ge80','sum'),ultima=('fecha','max'))
        agg['HR_previa24_edad_ultima_dias']=(date-agg.pop('ultima')).dt.days
        for col in agg:w.loc[ix,col]=ks.loc[ix].map(agg[col]).to_numpy()
    counts=['HR_previa24_n_mediciones','HR_previa24_n_mediciones_ge75','HR_previa24_n_mediciones_ge80']
    w[counts]=w[counts].fillna(0);w['HR_previa24_sin_mediciones']=(w.HR_previa24_n_mediciones==0).astype(float)
    for process in ['germinacion','apresorio']:
        for name,lags in [('rezagos1_19',range(1,20)),('rezagos20_24',range(20,25))]:
            vals=w[[f'{process}_termica_dmenos{k}' for k in lags]]
            w[process+'_'+name+'_n_observados']=vals.notna().sum(axis=1)
            w[process+'_'+name+'_fraccion_termica']=vals.mean(axis=1)
    return w

def initial_decision(p):return (np.asarray(p)>=.5).sum(axis=1)>=2

def targets(y,reference):return np.searchsorted(CODES,c.truth_codes(y,reference))

class VariationalStateNN:
    """Bayes by Backprop: q(w)=Normal(mu,softplus(rho)^2), priorN(0,1).

    Arquitectura3–8–4, tanh; likelihood categórica condicionada a decisión
    inicial conocida, con máscara TN/FN oTP/FP. KL analítica, Adam propio.
    """
    def __init__(self,seed=42):self.seed=seed
    def initialize(self,x):
        self.scale=StandardScaler().fit(x);rng=np.random.default_rng(self.seed)
        shapes=[(3,8),(8,),(8,4),(4,)]
        self.mu=[rng.normal(0,.15,s) if len(s)>1 else np.zeros(s) for s in shapes]
        self.rho=[np.full(s,-3.) for s in shapes];self.trace=[]
        return self
    def loss_grad(self,x,y,reference,eps,n):
        sigma=[np.logaddexp(0,r) for r in self.rho];weights=[m+s*e for m,s,e in zip(self.mu,sigma,eps)]
        w1,b1,w2,b2=weights;hidden=np.tanh(x@w1+b1);logits=hidden@w2+b2
        allowed=np.column_stack([~reference,~reference,reference,reference]);logits=np.where(allowed,logits,-1e12)
        logp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(logp)
        ce=-logp[np.arange(len(x)),y].mean();delta=p.copy();delta[np.arange(len(x)),y]-=1;delta/=len(x)
        dh=(delta@w2.T)*(1-hidden**2)
        gradients=[x.T@dh,dh.sum(axis=0),hidden.T@delta,delta.sum(axis=0)]
        kl=sum(.5*np.sum(m*m+s*s-1-2*np.log(s)) for m,s in zip(self.mu,sigma))
        gm=[grad+m/n for grad,m in zip(gradients,self.mu)]
        gr=[(grad*e+(s-1/s)/n)*expit(r) for grad,e,s,r in zip(gradients,eps,sigma,self.rho)]
        return float(ce+kl/n),gm+gr
    def fit(self,p,y,epochs=320,checkpoints=(80,160,320)):
        self.initialize(p);x=self.scale.transform(p);ref=initial_decision(p);target=targets(y,ref);rng=np.random.default_rng(self.seed)
        params=self.mu+self.rho;first=[np.zeros_like(v) for v in params];second=[np.zeros_like(v) for v in params];step=0;states={}
        for epoch in range(1,epochs+1):
            losses=[]
            for batch in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/512))):
                eps=[rng.normal(size=v.shape) for v in self.mu]
                loss,grad=self.loss_grad(x[batch],target[batch],ref[batch],eps,len(x));step+=1;losses.append(loss)
                for i,(v,dv) in enumerate(zip(params,grad)):
                    dv=np.clip(dv,-10,10);first[i]=.9*first[i]+.1*dv;second[i]=.999*second[i]+.001*dv*dv
                    v-=.005*(first[i]/(1-.9**step))/(np.sqrt(second[i]/(1-.999**step))+1e-8)
            if epoch%20==0:self.trace.append(dict(epoch=epoch,ELBO_negativa_media=float(np.mean(losses))))
            if epoch in checkpoints:states[epoch]=copy.deepcopy(self)
        return states
    def samples(self,p,n_samples=128):
        p=np.asarray(p,float);x=self.scale.transform(p);ref=initial_decision(p);rng=np.random.default_rng(20261026);out=[]
        allowed=np.column_stack([~ref,~ref,ref,ref]);sigma=[np.logaddexp(0,r) for r in self.rho]
        for _ in range(n_samples):
            w1,b1,w2,b2=[m+s*rng.normal(size=m.shape) for m,s in zip(self.mu,sigma)]
            logits=np.tanh(x@w1+b1)@w2+b2;logits=np.where(allowed,logits,-1e12)
            out.append(np.exp(logits-logsumexp(logits,axis=1,keepdims=True)))
        return np.asarray(out)
    def predict_details(self,p,n_samples=128):
        samples=self.samples(p,n_samples);mean=samples.mean(axis=0);presence_samples=samples[:,:,[0,2]].sum(axis=2)
        return pd.DataFrame(dict(P_FN_inicial=mean[:,0],P_TN_inicial=mean[:,1],P_TP_inicial=mean[:,2],P_FP_inicial=mean[:,3],
            P_correcto_inicial=mean[:,1]+mean[:,2],P_error_inicial=mean[:,0]+mean[:,3],
            probabilidad_roya=presence_samples.mean(axis=0),posterior_inferior95=np.quantile(presence_samples,.025,axis=0),
            posterior_superior95=np.quantile(presence_samples,.975,axis=0),desviacion_posterior=presence_samples.std(axis=0)))

def run(exchange=None):
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    d=g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv');m=pd.read_csv(g.OUT/'comunidades_por_observacion.csv')
    assert np.array_equal(d.fila_excel,m.fila_excel)
    split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';y=d[g.TARGET].to_numpy(float);yb=y>0
    old=joblib.load(b.OUT/'modelo.joblib');q=old['quantity'];history=humidity_history(d);w=features(d,q['bank'],history)
    previous=pd.read_csv(b.OUT/'entradas_OOF.csv',float_precision='round_trip');fold=previous.fold.to_numpy();sites=g.site_ids(d[tr]).to_numpy()
    oof=previous[['p_contexto','p_previos24','p_individual']].to_numpy();prob=np.zeros((len(d),3));models=old['models']
    for k in np.unique(fold):
        a=fold!=k;z=fold==k;assert not set(sites[a])&set(sites[z])
        rf=c.forest(42+int(k)).fit(w[tr].iloc[a],yb[tr][a]);oof[z,1]=rf.predict_proba(w[tr].iloc[z])[:,1];print('HR_OOF',int(k),flush=True)
    models[1]=c.forest().fit(w[tr],yb[tr]);prob[:,1]=models[1].predict_proba(w)[:,1]
    base_predictions=pd.read_csv(b.OUT/'predicciones.csv',float_precision='round_trip');assert np.array_equal(base_predictions.fila_excel,d.fila_excel)
    prob[:,0]=base_predictions.p_contexto;prob[:,2]=base_predictions.p_individual
    # Control Beta3 sobre las nuevas puntuaciones, antes de comparar con BNN.
    beta_models=[b.JointBeta3(k,20).fit(oof,yb[tr]) for k in [3,5]]
    beta_model=min(beta_models,key=lambda z:b.prior.probability_stats(yb[va],z.predict(prob[va]))['Brier']);p_beta=beta_model.predict(prob)
    beta_policy=min(b.prior.score_policies(yb[va],p_beta[va]),key=lambda v:(v['FP']+v['FN'],-v['exactitud_balanceada']))
    nets=VariationalStateNN().fit(oof,yb[tr]);candidates=[]
    for epoch,net in nets.items():
        p=net.predict_details(prob[va]).probabilidad_roya.to_numpy();quality=b.prior.probability_stats(yb[va],p)
        for row in b.prior.score_policies(yb[va],p):candidates.append(dict(epoch=epoch,**row,**quality))
    chosen=min(candidates,key=lambda v:(v['FP']+v['FN'],v['Brier'],-v['exactitud_balanceada']));net=nets[chosen['epoch']]
    selection=dict(red=chosen,control_beta=beta_policy,criterio='Mínimo FP+FN de validación, mismo costo por error; empate por Brier. Arquitectura3–8–4 fija, checkpoints80/160/320.',
        mejora_ambos_vs_beta=chosen['FP']<=beta_policy['FP'] and chosen['FN']<=beta_policy['FN'])
    g.csv(pd.DataFrame(candidates),OUT/'candidatos_validacion.csv');(OUT/'seleccion_antes_prueba.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    if exchange:exchange.validation_ready(selection)
    print('seleccion_congelada',selection,flush=True)
    details=net.predict_details(prob);p=details.probabilidad_roya.to_numpy();decision=p>=chosen['umbral'];reference=initial_decision(prob)
    comparison=[];quality=[]
    for tag,mask in [('validacion',va),('prueba',te)]:
        for name,score,cut in [('bosque_HR',prob[:,1],.5),('control_beta3_HR',p_beta,beta_policy['umbral']),('red_bayesiana_HR',p,chosen['umbral'])]:
            comparison.append(dict(conjunto=tag,modelo=name,**g.stats(yb[mask],score[mask]>=cut)))
            quality.append(dict(conjunto=tag,modelo=name,**b.prior.probability_stats(yb[mask],score[mask])))
    z=q['reg_representation'].transform(g.tabular(d,m,'combinadas'));reg_rows=[];pred_quantity={}
    for name in b.REGRESSIONS:
        pred_quantity[name]=np.where(decision,g.reg_predict(q['regressors'][name],name,z,d),0)
        for tag,mask in [('validacion',va),('prueba',te)]:reg_rows.append(dict(modelo=name,conjunto=tag,MAE=float(mean_absolute_error(y[mask],pred_quantity[name][mask])),RMSE=float(np.sqrt(mean_squared_error(y[mask],pred_quantity[name][mask])))))
    reg_chosen=min([v for v in reg_rows if v['conjunto']=='validacion'],key=lambda v:v['MAE'])['modelo']
    old.update(models=models,humidity_history=history,window_features=w.columns.tolist(),bnn=net,selection_HR=selection,beta_HR=beta_model,selected_regression=reg_chosen,
               humidity_available=True,humidity_simulated_available=False,humidity_source='observada_quincenal',reference_rule='mayoria de3 puntuaciones >=0.5')
    g.save_model(old,OUT/'modelo.joblib')
    rows=pd.concat([d[['fila_excel',g.TARGET]].reset_index(drop=True),details,b.outcomes(p,decision)],axis=1)
    rows['conjunto']=split;rows['presencia_predicha']=decision;rows['referencia_mayoria']=reference;rows['codigo_real_referencia']=c.truth_codes(yb,reference)
    rows['P_error_final']=np.where(decision,1-p,p);rows['P_correcto_final']=1-rows.P_error_final
    rows['requiere_revision']=(rows.posterior_inferior95<=chosen['umbral'])&(rows.posterior_superior95>=chosen['umbral'])
    for i,name in enumerate(['contexto','previos24_HR','individual']):rows['p_'+name]=prob[:,i]
    for name,value in pred_quantity.items():rows['incidencia_'+name]=value
    rows['incidencia_predicha']=pred_quantity[reg_chosen];g.csv(rows,OUT/'predicciones.csv')
    o=pd.DataFrame(oof,columns=['p_contexto','p_previos24_HR','p_individual']);o['fila_excel']=d.loc[tr,'fila_excel'].to_numpy();o['fold']=fold;o['presencia_real']=yb[tr];o['referencia_mayoria']=initial_decision(oof);o['codigo_real_referencia']=c.truth_codes(yb[tr],initial_decision(oof));g.csv(o,OUT/'entradas_OOF.csv')
    g.csv(pd.concat([d[['fila_excel']],w],axis=1),OUT/'antecedentes_HR_latencia.csv');g.csv(history,OUT/'historia_HR_real.csv');g.csv(pd.DataFrame(comparison),OUT/'comparacion.csv');g.csv(pd.DataFrame(quality),OUT/'calidad_probabilidades.csv');g.csv(pd.DataFrame(reg_rows),OUT/'dos_regresiones.csv');g.csv(pd.DataFrame(net.trace),OUT/'entrenamiento_BNN.csv')
    report=dict(estado='experimental_HR_real_no_simulada',diseno='Misma partición50/20/30 y cinco folds OOF por coordenadas. Bosques contexto/individual conservados; segundo bosque reentrenado con HR.',
        entradas_bosque24=w.columns.tolist(),seleccion=selection,comparacion=comparison,calidad_probabilidades=quality,regresiones=reg_rows,regresion_elegida=reg_chosen,
        red=dict(arquitectura=[3,8,4],metodo='Bayes by Backprop variacional diagonal Gaussiano',prior='Normal(0,1)',likelihood='Categórica condicionada a mayoría inicial; máscara de estados imposibles',muestras_posteriores=128,
            fuente='https://proceedings.mlr.press/v37/blundell15.html',entrada='Sólo3probabilidades OOF; escalado ajustado en train OOF.'),
        humedad=dict(tipo='Real, quincenal. No simulada ni diaria.',umbrales_exploratorios_usuario=[75,80],faltantes_previos='Se conservan NaN con indicador; conteos son mediciones, no días.'),
        latencia=dict(hipotesis_usuario_dias=[20,24],implementacion='Resumen térmico rezagado20–24 separado de1–19. No prueba colonización ni fecha de infección.',rango_no_universal=True),
        limites=['HR75/80 son hipótesis de comparación, no condiciones suficientes de infección ni sustitutos de mojado foliar.',
            'Los datos de HR históricos se ubican en fechas aproximadas1/16; no se interpolan a diario.',
            'La red aprende4estados relativos a la mayoría inicial. P(TP)+P(FN) determina la nueva presencia; se muestran riesgos relativos a decisión final por separado.',
            'Los intervalos son variacionales aproximados sobre pesos; no cobertura epidemiológica validada.',
            'La historia usa covariables pasadas del mismo sitio, sin incidencia actual ni pasada como entradas. Evalúa lugares nuevos con clima previo conocido, no lugares sin historia.',
            'Las redes geográficas usan covariables train y comunidad dependiente del periodo. Falta validación en fechas/localidades independientes.',
            'Muestreos17–24 en cero y unidad de incidencia sin verificar. Prueba reutilizada: resultados exploratorios.',
            'La latencia puede cambiar con hospedero y ambiente; no se fija biológicamente en24d.'])
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));write_report(report);return report

def predict(bundle,d):
    d=g.enrich(bundle['quantity'],d).reset_index(drop=True);q=bundle['quantity'];m=g.graph_transform(d,q['bank'],q['blocks']);m=b.extra_transform(d,m,bundle['extra_graphs'])
    w=features(d,q['bank'],bundle['humidity_history']);prob=np.column_stack([rf.predict_proba(f)[:,1] for rf,f in zip(bundle['models'],[m[b.CONTEXT],w,d[c.FOREST_FEATURES]])])
    details=bundle['bnn'].predict_details(prob);p=details.probabilidad_roya.to_numpy();a=p>=bundle['selection_HR']['red']['umbral'];out=pd.concat([details,b.outcomes(p,a)],axis=1)
    out['presencia_predicha']=a;out['referencia_mayoria']=initial_decision(prob)
    out['P_error_final']=np.where(a,1-p,p);out['P_correcto_final']=1-out.P_error_final
    out['requiere_revision']=(out.posterior_inferior95<=bundle['selection_HR']['red']['umbral'])&(out.posterior_superior95>=bundle['selection_HR']['red']['umbral'])
    for i,name in enumerate(['contexto','previos24_HR','individual']):out['p_'+name]=prob[:,i]
    z=q['reg_representation'].transform(g.tabular(d,m,'combinadas'))
    for name in b.REGRESSIONS:out['incidencia_'+name]=np.where(a,g.reg_predict(q['regressors'][name],name,z,d),0)
    out['incidencia_predicha']=out['incidencia_'+bundle['selected_regression']];out['estado']='experimental_HR_real_no_simulada'
    return out

def write_report(r):
    lines=['# Humedad relativa, rezagos20–24d y red neuronal bayesiana','',
    'Se añadió HR real continua y umbrales exploratorios75/80% al bosque de antecedentes. Se usan mediciones previas dentro de24d y su antigüedad; no se convierten registros quincenales en días húmedos. La HR actual también se conserva, por lo que ésta es una clasificación contemporánea, no pronóstico a24d.','',
    'La ventana térmica20–24d se separa de1–19d para contrastar la hipótesis del usuario. No se interpreta como inicio observado de colonización. Un experimento encontró latencia de21d en Caturra y37d en híbridos; la ventana no es universal: https://www.frontiersin.org/articles/10.3389/fpls.2017.02025/full . El agua libre es necesaria para germinación; HR alta por sí sola no demuestra infección: https://www.apsnet.org/edcenter/pdlessons/Pages/CoffeeRust.aspx .','',r['diseno'],'',
    '## Red bayesiana de corrección','',
    'Entradas: tres probabilidades de los bosques, estandarizadas en OOF train. Red3–8–4 con tanh; distribuciones Gaussianas variacionales en TODOS los pesos y sesgos, priorNormal(0,1), objetivo NLL+KL/N y Adam. Método Bayes by Backprop: https://proceedings.mlr.press/v37/blundell15.html . No es una MLP determinista renombrada ni un promedio de bosques.','',
    'Referencia inicial: mayoría de tres bosques a0.5. Las cuatro salidas son P(FN),P(TN),P(TP),P(FP) respecto a esa referencia. Estados incompatibles se enmascaran. P(roya)=P(FN)+P(TP), y un umbral de validación produce la decisión corregida. Los cuatro riesgos real/predicho finales se entregan aparte para evitar confundir corrección inicial y clasificación final.','',
    'Checkpoints80/160/320épocas y umbrales se seleccionaron por mínimo FP+FN de validación (costos iguales), con Brier para desempatar; luego se congeló antes de prueba. La inferencia promedia128sorteos de pesos y muestra dispersión e intervalo variacional. Minimizar la suma no garantiza que ambos errores bajen: se reportan por separado.','',
    '## Resultados','',pd.DataFrame(r['comparacion']).to_markdown(index=False),'',
    '## Probabilidades','',pd.DataFrame(r['calidad_probabilidades']).to_markdown(index=False),'',
    '## Dos regresiones después de presencia','',pd.DataFrame(r['regresiones']).to_markdown(index=False),'','Elegida por validación: '+r['regresion_elegida']+'. Coeficientes positivos y ACP previos conservados.','',
    '## Límites y siguientes pasos','']+['- '+s for s in r['limites']]+['',
    'Obtener HR diaria y mojado foliar para evaluar condiciones conjuntas; verificar ceros; validar en una nueva cohorte y comparar costos de omisión/inspección antes de promover esta ruta.']
    (OUT/'INFORME.md').write_text('\n'.join(lines))

if __name__=='__main__':
    from roya_humedad_bnn import run as canonical_run
    canonical_run()
