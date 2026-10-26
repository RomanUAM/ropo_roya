"""Bosque global de comunidades + bosque individual de clima/fenología.

Consenso empírico bayesiano Beta-Bernoulli por celdas conjuntas de puntuaciones.
No multiplica probabilidades ni supone independencia entre los dos bosques.
"""
import json,hashlib
import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss,log_loss,balanced_accuracy_score,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
import corrector_cuatro_estados as c

ROOT=c.ROOT
OUT=ROOT/'evaluacion_dos_bosques_bayes'
GLOBAL=['comunidad_topografia','comunidad_urbanistica','comunidad_condiciones15']
SIX=['lineal_multiple','log1p_lineal','polinomio_2','polinomio_3','ciclico','red_6_9_18_36_36']

class JointBetaConsensus:
    def __init__(self,bins=5,strength=20.):self.bins=bins;self.strength=strength
    def indices(self,p):
        p=np.asarray(p,float)
        return tuple(np.searchsorted(cuts,p[:,i],side='right') for i,cuts in enumerate(self.cuts_))
    def fit(self,p,y):
        p=np.asarray(p,float);y=np.asarray(y,float)
        self.cuts_=[]
        for col in p.T:
            cuts=np.unique(np.quantile(col,np.arange(1,self.bins)/self.bins))
            self.cuts_.append(cuts[(cuts>col.min())&(cuts<col.max())])
        shape=tuple(len(v)+1 for v in self.cuts_);self.n_=np.zeros(shape);self.k_=np.zeros(shape)
        ix=self.indices(p);np.add.at(self.n_,ix,1);np.add.at(self.k_,ix,y)
        # Bayes empírico: media previa de train y20 pseudo-observaciones.
        self.prior_mean_=float((y.sum()+.5)/(len(y)+1));self.a0_=self.strength*self.prior_mean_;self.b0_=self.strength*(1-self.prior_mean_)
        self.a_=self.a0_+self.k_;self.b_=self.b0_+self.n_-self.k_
        return self
    def predict(self,p):
        ix=self.indices(p);a=self.a_[ix];b=self.b_[ix]
        return a/(a+b)
    def details(self,p):
        ix=self.indices(p);a=self.a_[ix];b=self.b_[ix]
        return pd.DataFrame(dict(celda_global=ix[0],celda_individual=ix[1],observaciones_celda=self.n_[ix].astype(int),
            probabilidad_roya=a/(a+b),limite_inferior95=beta.ppf(.025,a,b),limite_superior95=beta.ppf(.975,a,b)))

def global_forest(seed=42):
    return make_pipeline(OneHotEncoder(handle_unknown='ignore',sparse_output=True),RandomForestClassifier(
        n_estimators=180,max_depth=12,min_samples_leaf=15,class_weight='balanced_subsample',n_jobs=2,random_state=seed))

def probability_stats(y,p):
    return dict(Brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,np.clip(p,1e-8,1-1e-8),labels=[False,True])))

def score_policies(y,p):
    return [dict(umbral=float(t),**c.g.stats(y,p>=t)) for t in np.arange(.025,.976,.025)]

def recommend(rows,baseline):
    good=[r for r in rows if r['FP']<=baseline['FP'] and r['FN']<=baseline['FN'] and (r['FP']<baseline['FP'] or r['FN']<baseline['FN']) and r['exactitud_balanceada']>=baseline['exactitud_balanceada']+.001]
    return max(good,key=lambda r:(r['exactitud_balanceada'],-r['FP'],-r['FN'])) if good else None

def error_probabilities(p,decision):
    decision=np.asarray(decision,bool);p=np.asarray(p,float)
    return dict(prob_falso_negativo=np.where(~decision,p,0),prob_falso_positivo=np.where(decision,1-p,0),
        prob_correcto=np.where(decision,p,1-p))

def agreement_table(pg,pi,y,tg,ti,model):
    rows=[]
    for a in [False,True]:
        for b in [False,True]:
            mask=((pg>=tg)==a)&((pi>=ti)==b);n=int(mask.sum());k=int(y[mask].sum())
            alpha=model.a0_+k;bet=model.b0_+n-k
            rows.append(dict(global_positivo=a,individual_positivo=b,registros_OOF=n,con_roya_OOF=k,
                probabilidad_roya=(alpha/(alpha+bet)),inferior95=float(beta.ppf(.025,alpha,bet)),superior95=float(beta.ppf(.975,alpha,bet))))
    return pd.DataFrame(rows)

def run(exchange=None):
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    def event(action,evidence):
        print(action,str(evidence)[:1100],flush=True)
        with (OUT/'eventos.jsonl').open('a') as s:s.write(json.dumps(dict(accion=action,evidencia=evidence),ensure_ascii=False)+'\n')
    d=c.g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(c.g.OUT/'comunidades_por_observacion.csv');assert np.array_equal(m.fila_excel,d.fila_excel)
    split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';y=d[c.g.TARGET].to_numpy(float);yb=y>0
    old=joblib.load(c.OUT/'modelo.joblib');individual=old['forest'];ti=old['forest_threshold']
    previous_oof=pd.read_csv(c.OUT/'objetivos_OOF.csv');assert np.array_equal(previous_oof.fila_excel,d.loc[tr,'fila_excel'])
    fold=previous_oof.fold.to_numpy();sites=c.g.site_ids(d[tr]).to_numpy();global_oof=np.zeros(tr.sum())
    for k in sorted(np.unique(fold)):
        a=fold!=k;b=fold==k;assert not set(sites[a])&set(sites[b])
        rf=global_forest(42+int(k)).fit(m.loc[tr,GLOBAL].iloc[a],yb[tr][a]);global_oof[b]=rf.predict_proba(m.loc[tr,GLOBAL].iloc[b])[:,1]
        event('global_OOF',dict(fold=int(k),registros=int(b.sum())))
    tg=float(max(np.arange(.05,.951,.025),key=lambda t:balanced_accuracy_score(yb[tr],global_oof>=t)))
    global_model=global_forest().fit(m.loc[tr,GLOBAL],yb[tr]);pg=global_model.predict_proba(m[GLOBAL])[:,1]
    pi=individual.predict_proba(d[c.FOREST_FEATURES])[:,1]
    oof=np.column_stack([global_oof,previous_oof.probabilidad_OOF]);prob=np.column_stack([pg,pi])
    bayes={};resolutions=[]
    for bins in [3,5,8]:
        name='beta_conjunta_'+str(bins);model=JointBetaConsensus(bins=bins,strength=20).fit(oof,yb[tr]);bayes[name]=model
        val=model.predict(prob[va]);resolutions.append(dict(modelo=name,celdas=int(model.n_.size),**probability_stats(yb[va],val)))
    selected=min(resolutions,key=lambda r:r['Brier'])['modelo'];model=bayes[selected];p=model.predict(prob)
    val_forests=[dict(nombre='global',**c.g.stats(yb[va],pg[va]>=tg)),dict(nombre='individual',**c.g.stats(yb[va],pi[va]>=ti))]
    reference=max(val_forests,key=lambda r:r['exactitud_balanceada']);policies=score_policies(yb[va],p[va]);balanced=max(policies,key=lambda r:(r['exactitud_balanceada'],-r['FP'],-r['FN']))
    conservative=recommend(policies,reference)
    selection=dict(modelo=selected,criterio_probabilidades='Menor Brier de validación entre3 resoluciones conjuntas; priors y parámetros ajustados sólo con OOF train.',
        referencia=reference,adoptar_clasificacion=conservative is not None,politica_recomendada=conservative,politica_balanceada=balanced,
        criterio_clasificacion='Recomendar consenso sólo si no aumenta FP/FN del mejor bosque de validación y mejora exactitud balanceada>=0.001; si no, conservar ese bosque como decisión y mostrar posterior bayesiana.')
    c.g.csv(pd.DataFrame(policies),OUT/'politicas_validacion.csv');c.g.csv(pd.DataFrame(resolutions),OUT/'resoluciones_validacion.csv')
    (OUT/'seleccion_antes_prueba.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    if exchange:exchange.validation_ready(selection)
    event('seleccion_congelada',selection)
    glob=pg>=tg;ind=pi>=ti;bal=p>=balanced['umbral'];recommended=p>=conservative['umbral'] if conservative else (glob if reference['nombre']=='global' else ind)
    table_oof=d.loc[tr,['fila_excel','Coordenada_X','Coordenada_Y']].copy();table_oof['fold']=fold;table_oof['p_global']=global_oof;table_oof['p_individual']=previous_oof.probabilidad_OOF.to_numpy();table_oof['presencia_real']=yb[tr]
    c.g.csv(table_oof,OUT/'entradas_consenso_OOF.csv');c.g.csv(agreement_table(global_oof,previous_oof.probabilidad_OOF.to_numpy(),yb[tr],tg,ti,model),OUT/'probabilidades_por_acuerdo_train.csv')
    comparison=[];quality=[];predictions=[];reliability=[]
    for tag,mask in [('validacion',va),('prueba',te)]:
        for name,decision in [('bosque_global',glob),('bosque_individual',ind),('consenso_balanceado',bal),('recomendado',recommended)]:comparison.append(dict(conjunto=tag,modelo=name,**c.g.stats(yb[mask],decision[mask])))
        for name,score in [('bosque_global',pg),('bosque_individual',pi),('consenso_bayesiano',p)]:
            quality.append(dict(conjunto=tag,modelo=name,**probability_stats(yb[mask],score[mask])))
            bins=np.minimum((score[mask]*10).astype(int),9)
            for k in np.unique(bins):
                a=bins==k;reliability.append(dict(conjunto=tag,modelo=name,bin=int(k),registros=int(a.sum()),probabilidad_media=float(score[mask][a].mean()),fraccion_real_con_roya=float(yb[mask][a].mean())))
        t=d.loc[mask,['fila_excel','Municipio','Parcela','Mes_registrado','Quincena','Coordenada_X','Coordenada_Y',c.g.TARGET]].copy().reset_index(drop=True)
        details=model.details(prob[mask]);t=pd.concat([t,details],axis=1);t['conjunto']=tag;t['p_global']=pg[mask];t['p_individual']=pi[mask]
        t['presencia_global']=glob[mask];t['presencia_individual']=ind[mask];t['presencia_consenso_balanceado']=bal[mask];t['presencia_predicha']=recommended[mask]
        for scope,decision in [('global',glob[mask]),('individual',ind[mask]),('consenso',recommended[mask])]:
            for key,value in error_probabilities(p[mask],decision).items():t[key+'_'+scope]=value
        t['codigo_error_global_real']=c.truth_codes(yb[mask],glob[mask]);t['codigo_error_global_estimado']=np.where(glob[mask],np.where(recommended[mask],1,2),np.where(recommended[mask],-1,0))
        predictions.append(t)
    c.g.csv(pd.DataFrame(comparison),OUT/'comparacion.csv');c.g.csv(pd.DataFrame(quality),OUT/'calidad_probabilidades.csv');c.g.csv(pd.DataFrame(reliability),OUT/'calibracion.csv')
    quantity=old['quantity'];quantity['regressors']={k:v for k,v in quantity['regressors'].items() if k in SIX};assert len(quantity['regressors'])==6
    z=quantity['reg_representation'].transform(c.g.tabular(d,m,'combinadas'));reg_rows=[]
    for name,reg in quantity['regressors'].items():
        for tag,mask in [('validacion',va),('prueba',te)]:
            q=np.where(recommended[mask],c.g.reg_predict(reg,name,z[mask],d[mask]),0);reg_rows.append(dict(modelo=name,conjunto=tag,MAE=float(mean_absolute_error(y[mask],q)),RMSE=float(np.sqrt(mean_squared_error(y[mask],q)))))
    chosen=min([v for v in reg_rows if v['conjunto']=='validacion'],key=lambda v:v['MAE'])['modelo'];quantity['selected_regression']=chosen
    for t in predictions:
        mask=split==t.conjunto.iloc[0];t['incidencia_predicha']=np.where(recommended[mask],c.g.reg_predict(quantity['regressors'][chosen],chosen,z[mask],d[mask]),0)
    c.g.csv(pd.concat(predictions),OUT/'predicciones.csv');c.g.csv(pd.DataFrame(reg_rows),OUT/'seis_regresiones.csv')
    bundle=dict(global_model=global_model,individual_model=individual,global_threshold=tg,individual_threshold=ti,
        bayes_models=bayes,selected_bayes=selected,selection=selection,quantity=quantity)
    report=dict(diseno='Mismas coordenadas50/20/30; cinco folds OOF comunes a ambos bosques. El bosque individual original se reutiliza sin cambios.',
        global_features=GLOBAL,individual_features=c.FOREST_FEATURES,umbral_global=tg,umbral_individual=ti,
        bayes='Bayes empírico Beta-Bernoulli en celdas conjuntas de ambas puntuaciones OOF; media previa de train y fuerza20. No asume independencia entre los bosques.',
        formula='theta_c | datos ~ Beta(alpha0+k_c,beta0+n_c-k_c); E(theta_c|datos)=(alpha0+k_c)/(alpha0+beta0+n_c)',
        seleccion=selection,comparacion=comparison,calidad_probabilidades=quality,resoluciones_validacion=resolutions,regresiones=reg_rows,regresion_elegida=chosen,
        acuerdos_train=agreement_table(global_oof,previous_oof.probabilidad_OOF.to_numpy(),yb[tr],tg,ti,model).to_dict('records'),
        limites=['Prueba histórica consultada muchas veces: estudio exploratorio, no validación prospectiva.',
            'Muestreos17–24 son ceros y falta confirmar si son ausencia real o captura incompleta; unidad de incidencia pendiente.',
            'P(error|pG,pI) es estimación del modelo, no conocimiento de si ese caso está infectado. Para verificar se necesita observación real.',
            'Los intervalos Beta son condicionales al modelo y al binning; las observaciones repetidas por planta/sitio están correlacionadas y los intervalos no representan cobertura espacial/temporal independiente.',
            'Las comunidades15d incluyen el periodo en su identificador; esta evaluación usa fechas conocidas. Para fechas nuevas se deben reconstruir y validar redes y mapeo de comunidades.',
            'Las condiciones15d son aptitud térmica simulada con umbral0.9, no infección ni mojado foliar observados.',
            'Una mejora del Brier no demuestra por sí sola calibración perfecta; revisar curva de fiabilidad y validación nueva.',
            'Las seis regresiones conservan coeficientes entrenados sobre positivos previos; se reevalúa su puerta de presencia.'])
    balanced_quantity(report,bundle)
    c.g.save_model(bundle,OUT/'modelo.joblib')
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));make_report(report);plot(report)
    event('finalizado',dict(modelo_bayes=selected,consenso_adoptado=selection['adoptar_clasificacion'],regresion=chosen))
    return report

def predict(bundle,d,variant='recomendado'):
    d=c.g.enrich(bundle['quantity'],d).reset_index(drop=True);m=c.g.graph_transform(d,bundle['quantity']['bank'],bundle['quantity']['blocks'])
    pg=bundle['global_model'].predict_proba(m[GLOBAL])[:,1];pi=bundle['individual_model'].predict_proba(d[c.FOREST_FEATURES])[:,1]
    model=bundle['bayes_models'][bundle['selected_bayes']];details=model.details(np.column_stack([pg,pi]));p=details.probabilidad_roya.to_numpy();selection=bundle['selection']
    glob=pg>=bundle['global_threshold'];ind=pi>=bundle['individual_threshold']
    if variant=='global':decision=glob
    elif variant=='individual':decision=ind
    elif variant=='balanceado':decision=p>=selection['politica_balanceada']['umbral']
    elif selection['politica_recomendada']:decision=p>=selection['politica_recomendada']['umbral']
    else:decision=glob if selection['referencia']['nombre']=='global' else ind
    quantity=bundle['quantity'];name=quantity.get('selected_regression_balanceado',quantity['selected_regression']) if variant=='balanceado' else quantity['selected_regression'];z=quantity['reg_representation'].transform(c.g.tabular(d,m,'combinadas'))
    result=pd.DataFrame(dict(probabilidad_global=pg,probabilidad_individual=pi,probabilidad_consenso=p,presencia_global=glob,presencia_individual=ind,observaciones_celda=details.observaciones_celda,
        limite_inferior95=details.limite_inferior95,limite_superior95=details.limite_superior95,presencia_predicha=decision,
        probabilidad_error_estimado=np.where(decision,1-p,p),probabilidad_acierto_estimado=np.where(decision,p,1-p),
        grupo_final=m.comunidad_condiciones15,incidencia_predicha=np.where(decision,c.g.reg_predict(quantity['regressors'][name],name,z,d),0),estado=np.where(decision,'cantidad_estimada','sin_roya')))
    for scope,base in [('global',glob),('individual',ind)]:
        for key,value in error_probabilities(p,base).items():result[key+'_'+scope]=value
    return result

def balanced_quantity(report,bundle):
    d=c.g.assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    m=pd.read_csv(c.g.OUT/'comunidades_por_observacion.csv');pred=pd.read_csv(OUT/'predicciones.csv').set_index('fila_excel');quantity=bundle['quantity']
    z=quantity['reg_representation'].transform(c.g.tabular(d,m,'combinadas'));rows=[]
    for name,model in quantity['regressors'].items():
        for tag in ['validacion','prueba']:
            mask=m.conjunto.to_numpy()==tag;gate=pred.loc[d.loc[mask,'fila_excel'],'presencia_consenso_balanceado'].to_numpy()
            q=np.where(gate,c.g.reg_predict(model,name,z[mask],d[mask]),0);rows.append(dict(modelo=name,conjunto=tag,MAE=float(mean_absolute_error(d.loc[mask,c.g.TARGET],q)),RMSE=float(np.sqrt(mean_squared_error(d.loc[mask,c.g.TARGET],q)))))
    chosen=min([v for v in rows if v['conjunto']=='validacion'],key=lambda v:v['MAE'])['modelo'];quantity['selected_regression_balanceado']=chosen
    report['regresiones_consenso_balanceado']=rows;report['regresion_elegida_consenso_balanceado']=chosen
    for tag in ['validacion','prueba']:
        mask=m.conjunto.to_numpy()==tag;ix=d.loc[mask,'fila_excel'];gate=pred.loc[ix,'presencia_consenso_balanceado'].to_numpy();q=np.where(gate,c.g.reg_predict(quantity['regressors'][chosen],chosen,z[mask],d[mask]),0)
        pred.loc[ix,'incidencia_consenso_balanceado']=q
    c.g.csv(pred.reset_index(),OUT/'predicciones.csv');c.g.csv(pd.DataFrame(rows),OUT/'seis_regresiones_consenso_balanceado.csv')

def make_report(r):
    lines=['# Dos bosques y consenso bayesiano','',
        'Bosque global: comunidades topográfica, urbanística y condiciones15d. Bosque individual: seis variables climáticas reales y seis etapas fenológicas. No se usan redes neuronales para clasificar en esta ruta.','',r['diseno'],'',
        '## Matrices de clasificación','',pd.DataFrame(r['comparacion']).to_markdown(index=False),'',
        'Consenso adoptado como clasificación recomendada: **'+str(r['seleccion']['adoptar_clasificacion'])+'**. '+r['seleccion']['criterio_clasificacion'],'',
        'La variante balanceada siempre aplica el posterior bayesiano; la recomendada conserva el mejor bosque si el consenso no mejora simultáneamente sus FP/FN en validación.','',
        '## Probabilidades','',r['bayes'],'',r['formula'],'',
        'Se aprende la distribución conjunta de las dos puntuaciones; no se multiplican probabilidades marginales. Los cuantiles que definen las celdas y los conteos se ajustan sólo con OOF train. La resolución se elige por Brier de validación, y luego se fija el umbral sin usar prueba.','',
        'Si el consenso clasifica negativo: P(FN|pG,pI)=p. Si clasifica positivo: P(FP|pG,pI)=1−p. La probabilidad de acierto es el complemento. Esto estima riesgo; no verifica el diagnóstico real. La misma cuenta se informa respecto a cada bosque individualmente.','',
        pd.DataFrame(r['calidad_probabilidades']).to_markdown(index=False),'',
        '## Acuerdos y desacuerdos de los bosques','',pd.DataFrame(r['acuerdos_train']).to_markdown(index=False),'',
        'Probabilidades agrupadas aprendidas con OOF train; cada observación recibe además su posterior por celda de puntuaciones, más detallado que estas cuatro combinaciones binarias.','',
        '## Seis regresiones después de la decisión recomendada','',pd.DataFrame(r['regresiones']).to_markdown(index=False),'',
        'Elegida por MAE de validación: '+r['regresion_elegida']+'. La regresión neuronal de cantidad se mantiene entre las seis familias; las redes neuronales de clasificación no forman parte de este consenso.','',
        '## Límites','']+['- '+v for v in r['limites']]+['',
        'Referencias: https://mc-stan.org/docs/functions-reference/beta-binomial-distribution.html ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html ; https://scikit-learn.org/stable/modules/calibration.html']
    if r.get('regresiones_consenso_balanceado'):
        lines+=['','## Seis regresiones con consenso bayesiano activo','',pd.DataFrame(r['regresiones_consenso_balanceado']).to_markdown(index=False),'',
            'Familia seleccionada por validación para el consenso activo: '+r['regresion_elegida_consenso_balanceado']+'. La clasificación conservadora es una comparación dentro de esta rama, no una recomendación de sustituir mejores perfiles históricos.']
    (OUT/'INFORME.md').write_text('\n'.join(lines))

def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[v for v in r['comparacion'] if v['conjunto']=='prueba'];fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    for ax,v in zip(axes,rows):
        a=np.array([[v['TN'],v['FP']],[v['FN'],v['TP']]]);ax.imshow(a,cmap='Blues');ax.set_title(v['modelo'].replace('_',' '),fontsize=10);ax.set_xticks([0,1],['Sin roya','Con roya']);ax.set_yticks([0,1],['Sin roya','Con roya']);ax.set_xlabel('Predicción');ax.set_ylabel('Observación')
        for (i,j),n in np.ndenumerate(a):ax.text(j,i,str(n),ha='center',va='center',fontsize=13,color='white' if n>a.max()/2 else 'black')
    fig.suptitle('Dos bosques y consenso · prueba por coordenadas');fig.savefig(OUT/'matrices_confusion.png',dpi=160);plt.close(fig)
    reliability=pd.read_csv(OUT/'calibracion.csv');fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True);ax.plot([0,1],[0,1],'--',color='gray')
    for name,g in reliability[reliability.conjunto=='prueba'].groupby('modelo'):ax.plot(g.probabilidad_media,g.fraccion_real_con_roya,'o-',label=name.replace('_',' '))
    ax.set(xlabel='Probabilidad media predicha',ylabel='Fracción observada con roya',title='Fiabilidad en prueba exploratoria',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8);fig.savefig(OUT/'fiabilidad.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    from dos_bosques_consenso_bayes import run as canonical_run
    canonical_run()
