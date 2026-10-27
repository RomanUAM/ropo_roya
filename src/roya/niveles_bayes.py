"""Cuatro arquitecturas multiclase y mezclador inverso Dirichlet/categórico."""
from pathlib import Path
import json,warnings
import numpy as np,pandas as pd,joblib
from scipy.special import gammaln,softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,accuracy_score,f1_score,cohen_kappa_score,mean_absolute_error,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.fusion_modal import PROJECT,ModalInputs,modal_frames,BRANCHES
from roya.variables import calendar
from roya.modelos.genetica_ordinal import GeneticOrdinal3
from roya.comunidades_regresion import OUT as COMMUNITY_OUT,design_frame
from roya.poisson_modal import audit_split
from roya.registro import write,Registry
OUT=PROJECT/'runs/niveles_bayes'


def labels(y):
    a=np.asarray(y,float)
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Incidencia inválida')
    return np.ceil(a/3).astype(int)


def pad_prob(est,x,nclasses):
    p=est.predict_proba(x);out=np.zeros((len(x),nclasses));out[:,np.asarray(est.classes_,int)]=p;return out


def frames(q,d,m,w):
    f=modal_frames(q,d,m,w);c=calendar(d)
    for view in f.values():
        for col in c:
            if col not in view:view[col]=c[col].to_numpy()
    return f


def contexts(q,d,m):
    df=design_frame(q,d,pd.read_csv(COMMUNITY_OUT/'comunidades_por_sitio_fecha.csv'));return pd.DataFrame({'topografia_urbana':df.comunidad_topografia.astype(str)+'|'+m.comunidad_urbanistica.astype(str).to_numpy(),'simulacion':df.comunidad_simulacion.astype(str),'excel':df.comunidad_excel.astype(str)})


class InverseBayes:
    """Densidades Dirichlet de probas y tablas categóricas, posterior templado."""
    def fit(self,p,y,context):
        p=self.clean(p);y=np.asarray(y,int);self.classes_=np.arange(p.shape[2]);k=len(self.classes_);count=np.bincount(y,minlength=k);self.prior_=(count+.5)/(len(y)+.5*k);self.alpha_=[]
        for j in range(p.shape[1]):
            allmean=p[:,j].mean(axis=0);allvar=p[:,j].var(axis=0);alphas=[]
            for c in self.classes_:
                rows=p[y==c,j];n=len(rows);a=n/(n+20);mu=a*(rows.mean(axis=0) if n else allmean)+(1-a)*allmean;variance=a*(rows.var(axis=0) if n else allvar)+(1-a)*allvar;concentration=np.clip(np.sum(mu*(1-mu))/max(float(variance.sum()),1e-6)-1,2,200);alphas.append(np.maximum(mu*concentration,.02))
            self.alpha_.append(np.array(alphas))
        self.tables_={}
        for col in context:
            values=context[col].astype(str);levels=sorted(values.unique());table={}
            for v in levels:table[v]=(np.bincount(y[values==v],minlength=k)+1)/(count+len(levels)+1)
            self.tables_[col]=(table,1/(count+len(levels)+1))
        return self
    @staticmethod
    def clean(p):
        p=np.clip(np.asarray(p,float),1e-6,1)
        if p.ndim!=3 or not np.isfinite(p).all():raise ValueError('Probabilidades inválidas')
        return p/p.sum(axis=2,keepdims=True)
    def evidence(self,p,context):
        p=self.clean(p);out=np.zeros((len(p),len(self.classes_)));g=np.zeros_like(out)
        for j,a in enumerate(self.alpha_):out+=np.log(p[:,j])@(a-1).T+gammaln(a.sum(axis=1))-gammaln(a).sum(axis=1)
        for col,(table,unknown) in self.tables_.items():g+=np.log(np.vstack([table.get(v,unknown) for v in context[col].astype(str)]))
        return out/len(self.alpha_),g/max(len(self.tables_),1)
    def predict_proba(self,p,context,temperature=1.,context_weight=1.):
        a,b=self.evidence(p,context);return softmax(np.log(self.prior_)+temperature*a+context_weight*b,axis=1)


def metrics(y,p,k):
    pred=p.argmax(axis=1);one=np.eye(k)[y];return dict(n=len(y),exactitud=float(accuracy_score(y,pred)),F1_macro=float(f1_score(y,pred,labels=np.arange(k),average='macro',zero_division=0)),F1_ponderado=float(f1_score(y,pred,labels=np.arange(k),average='weighted',zero_division=0)),log_loss=float(log_loss(y,p,labels=np.arange(k))),Brier_multiclase=float(np.mean(np.sum((p-one)**2,axis=1))),kappa_cuadratica=float(cohen_kappa_score(y,pred,labels=np.arange(k),weights='quadratic')),MAE_clases=float(mean_absolute_error(y,pred)))


def base(family,seed,config):
    if family=='RF':return RandomForestClassifier(n_estimators=config['trees'],max_depth=12,min_samples_leaf=15,class_weight='balanced_subsample',random_state=seed,n_jobs=2)
    if family=='LR':return LogisticRegression(C=1,class_weight='balanced',solver='newton-cholesky',max_iter=1000,tol=1e-6,random_state=seed)
    return GeneticOrdinal3(config['gp_population'],config['gp_generations'],seed)


def run(out=None):
    out=Path(out or OUT);out.mkdir(parents=True,exist_ok=True);threadpool_limits(2);q=load();d,m,w,_=q.data();audit=audit_split(q,d,m);y=labels(d[q.g.TARGET]);k=int(y.max())+1;split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';views=frames(q,d,m,w);ctx=contexts(q,d,m);config=json.loads((PROJECT/'configs/niveles_bayes.json').read_text());count=pd.crosstab(y,m.conjunto).reindex(np.arange(k),fill_value=0);count.index.name='clase';count.to_csv(out/'conteo_clases.csv');d[['fila_excel',q.g.TARGET,'Mes_registrado','Quincena']].assign(clase_roya=y,conjunto=split).to_csv(out/'etiquetas_por_fila.csv',index=False);write(out/'etiquetado.json',dict(regla='0→0; 0<y<=3→1; 3<y<=6→2; ...; ceil(y/3)',clases_observadas=k,rangos=[dict(clase=c,minimo=0 if c==0 else 3*(c-1)+1,maximo=3*c) for c in range(k)],nota='Datos actuales enteros. Para decimales los intervalos son abiertos a izquierda y cerrados a derecha; no se redondea y. Clases >36 sin soporte no se predicen.'))
    folds=pd.read_csv(PROJECT/'runs/calendario_r2/ACP/OOF_LR.csv');assert np.array_equal(folds.fila_excel,d.loc[tr,'fila_excel']);fold=folds.fold.to_numpy();groups=q.g.site_ids(d).loc[tr].reset_index(drop=True);allp={};alloof={};registry=Registry(out,'clasificacion_niveles_bayes_v1')
    protocol=dict(config=config,particion=audit,etiquetas='13 clases observadas 0..12 con anchos de tres puntos porcentuales; clase extrema sólo 3 train. Sin reinterpretar porcentajes como conteos.',vistas='Topografía+coordenadas+altitud+urbanismo; simulación/antecedentes; Excel real+fenología+variedades. Calendario en las tres. ModalInputs train-only: normalización y ACP de continuas.',OOF='5 folds por coordenadas compartidos; preprocesado y modelos nuevos en cada fold, sin reutilizar pesos binarios. Comunidades no supervisadas previas train de todo el año: limitación histórica.',GP='Tres modelos GP ordinal de árboles <=3 niveles; fitness MSE ordinal balanceado+parsimonia, probabilidades por densidad gaussiana del score condicional a clase con suavizado20. No es el GP binario original sin cambios.',mezclador='Bayes inverso aproximado: P(clase|3 vectores de probabilidad,3 grupos) proporcional a prior×densidades Dirichlet condicionales×verosimilitudes categóricas. Parámetros de clase ajustados en OOF train, suavizado y concentraciones acotadas.',contextos='Grupo topográfico unido a comunidad urbanística; grupo climático simulado; grupo climático Excel. Sin incidencia de vecinos.',independencia='Hipótesis aproximada entre expertos y grupos correlacionados; log-verosimilitudes promediadas y templadas. No se afirma calibración automática.',seleccion='Temperatura y peso de grupos sólo por menor log-loss validación. MIXTA: mejor vista de cada familia por log-loss OOF train. Prueba no elige arquitectura.',evaluacion='Matriz de confusión 13x13, F1 macro/por clase, log-loss, Brier y error ordinal; exactitud mayoritaria no basta. Test histórico reutilizado, exploratorio.')
    write(out/'protocolo.json',protocol);registry.send('datos','estadistica','reetiquetar_intervalos_tres',dict(conteos=count.to_dict()))
    for family in ['RF','LR','GP']:
        family_p=[];family_oof=[]
        for j,name in enumerate(BRANCHES):
            folder=out/family;folder.mkdir(exist_ok=True);file=folder/(name+'.joblib');probs=folder/(name+'_probas.npz');view=views[name]
            if file.exists() and probs.exists():
                with np.load(probs) as z:family_p.append(z['p']);family_oof.append(z['oof'])
                continue
            oof=np.zeros((tr.sum(),k));trainframe=view.loc[tr].reset_index(drop=True)
            for f in sorted(set(fold)):
                a=fold!=f;b=fold==f;assert not set(groups[a])&set(groups[b]);pre=ModalInputs().fit(q,trainframe.loc[a]);est=base(family,config['seed']+int(f),config)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');est.fit(pre.transform(trainframe.loc[a]),y[tr][a])
                if any(issubclass(z.category,ConvergenceWarning) for z in caught):raise RuntimeError('No convergió '+family+' '+name)
                oof[b]=pad_prob(est,pre.transform(trainframe.loc[b]),k);print('niveles',family,name,int(f),flush=True)
            pre=ModalInputs().fit(q,view.loc[tr]);est=base(family,config['seed'],config).fit(pre.transform(view.loc[tr]),y[tr]);p=pad_prob(est,pre.transform(view),k);joblib.dump(dict(preprocess=pre,model=est),file,compress=3);np.savez_compressed(probs,p=p,oof=oof);family_p.append(p);family_oof.append(oof)
        allp[family]=np.stack(family_p,axis=1);alloof[family]=np.stack(family_oof,axis=1)
    entries=[];scores=[]
    for family in ['RF','GP','LR']:
        qualities=[float(log_loss(y[tr],alloof[family][:,j],labels=np.arange(k))) for j in range(3)];j=int(np.argmin(qualities));entries.append(dict(familia=family,vista=BRANCHES[j],indice=j,log_loss_OOF=qualities[j]));scores.extend(dict(familia=family,vista=BRANCHES[z],log_loss_OOF=qualities[z]) for z in range(3))
    allp['MIXTA']=np.stack([allp[e['familia']][:,e['indice']] for e in entries],axis=1);alloof['MIXTA']=np.stack([alloof[e['familia']][:,e['indice']] for e in entries],axis=1);write(out/'entradas_mixta.json',entries);rows=[];selections={};predictions={};mats={}
    for family in ['RF','LR','GP','MIXTA']:
        bayes=InverseBayes().fit(alloof[family],y[tr],ctx.loc[tr]);candidates=[];ev,eg=bayes.evidence(allp[family][va],ctx.loc[va]);prior=np.log(bayes.prior_)
        for t in [0.,.05,.1,.25,.5,1.]:
            for g in [0.,.25,.5,1.]:
                p=softmax(prior+t*ev+g*eg,axis=1);candidates.append(dict(temperatura=t,peso_grupos=g,**metrics(y[va],p,k)))
        chosen=min(candidates,key=lambda r:(r['log_loss'],-r['F1_macro']));selections[family]=chosen;write(out/(family+'_seleccion_antes_prueba.json'),chosen);joblib.dump(dict(bayes=bayes,selection=chosen,entries=entries if family=='MIXTA' else None),out/(family+'_mezclador.joblib'),compress=3);pd.DataFrame(candidates).to_csv(out/(family+'_validacion.csv'),index=False)
        p=bayes.predict_proba(allp[family],ctx,chosen['temperatura'],chosen['peso_grupos']);predictions[family]=p;np.savez_compressed(out/(family+'_posterior.npz'),p=p,fila_excel=d.fila_excel.to_numpy(),y=y,conjunto=split)
        for tag,mask in [('validacion',va),('prueba',te)]:
            rows.append(dict(arquitectura=family,conjunto=tag,**metrics(y[mask],p[mask],k)));matrix=confusion_matrix(y[mask],p[mask].argmax(axis=1),labels=np.arange(k));pd.DataFrame(matrix,index=np.arange(k),columns=np.arange(k)).to_csv(out/(family+'_matriz_'+tag+'.csv'));mats[family+'_'+tag]=matrix.tolist()
    write(out/'seleccion_antes_prueba.json',selections);pd.DataFrame(rows).to_csv(out/'comparacion.csv',index=False);report=dict(protocolo=protocol,selecciones=selections,mixta=entries,resultados=rows,matrices=mats);write(out/'resultados.json',report);registry.send('auditoria','memoria','intervalos_no_son_cantidad_exacta',rows);registry.close();return report


def predict(request,family='RF',out=None):
    if family not in ['RF','LR','GP','MIXTA']:raise ValueError('Arquitectura desconocida')
    out=Path(out or OUT);q=load();d,_=q.old.registered_data(request);m=pd.read_csv(q.old.OUT/'comunidades_por_fila.csv').set_index('fila_excel').loc[d.fila_excel].reset_index();ref=q.old.load_reference();w=q.h.features(d,ref['quantity']['bank'],ref['humidity_history']);f=frames(q,d,m,w);ctx=contexts(q,d,m);bundle=joblib.load(out/(family+'_mezclador.joblib'));bayes=bundle['bayes'];entries=bundle['entries'] if family=='MIXTA' else [dict(familia=family,vista=name) for name in BRANCHES];probs=[]
    for entry in entries:
        b=joblib.load(out/entry['familia']/(entry['vista']+'.joblib'));probs.append(pad_prob(b['model'],b['preprocess'].transform(f[entry['vista']]),len(bayes.classes_)))
    s=bundle['selection'];p=bayes.predict_proba(np.stack(probs,axis=1),ctx,s['temperatura'],s['peso_grupos']);label=p.argmax(axis=1);result=pd.DataFrame({'fila_excel':d.fila_excel.to_numpy(),'clase_roya':label,'limite_inferior':np.where(label==0,0,3*(label-1)+1),'limite_superior':label*3,'probabilidad_clase':p.max(axis=1),'probabilidad_presencia':1-p[:,0],'presencia_por_masa':(1-p[:,0])>=.5})
    for k in range(p.shape[1]):result['p_clase_'+str(k)]=p[:,k]
    return result
