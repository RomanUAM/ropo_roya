"""Mahalanobis -> Louvain multicapa -> bosque -> rescate -> filtro -> cantidad.

Redes ajustadas exclusivamente en ubicaciones train. Asignación inductiva por
suma de pesos hacia comunidades train; nunca se recalculan con prueba.
"""
import json, warnings, hashlib, os, zlib
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, save_npz
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits
from modelo_roya_50_20_30 import assemble, inputs, TERRAIN, GEO
from roya_dos_etapas import TARGET
from probar_arbitro_logistico import cycle
from programacion_genetica_roya import SymbolicRegressor3

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evaluacion_redes_granulares'
STAGES=['Brotacion','Floracion','Amarre','Lechoso','Consistente','Maduro']
LAYERS=['topografia','simulada','condiciones15','ubicacion','urbanistica']
REGRESSIONS=['lineal_multiple','log1p_lineal','polinomio_2','polinomio_3','ciclico','red_6_9_18_36_36','genetica_3_niveles']

def csv(d,path):
    path=Path(path);tmp=path.with_suffix('.tmp');d.to_csv(tmp,index=False);tmp.replace(path)

def save_model(bundle,path):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        joblib.dump(bundle,stream,compress=3);stream.flush();os.fsync(stream.fileno())
    # Verificar final del flujo, además de deserializar, antes de promover.
    decoder=zlib.decompressobj();decoder.decompress(tmp.read_bytes())
    if not decoder.eof:raise IOError('Serialización incompleta del modelo; se conserva la versión anterior.')
    joblib.load(tmp);tmp.replace(path)

def site_ids(d):
    return pd.Series([f'{x:.3f}|{y:.3f}' for x,y in zip(d.Coordenada_X,d.Coordenada_Y)],index=d.index)

def periods(d):
    q=d.Quincena.astype(str).str.extract(r'(\d+)')[0].astype(int)
    return pd.to_datetime(d.Mes_registrado).dt.strftime('%Y-%m')+'-'+q.astype(str)

def partition(d):
    u=d.assign(sitio=site_ids(d)).drop_duplicates('sitio')[['sitio','Municipio']]
    rng=np.random.default_rng(42);a=[]
    for _,g in u.groupby('Municipio',sort=True):
        g=g.copy();perm=rng.permutation(len(g));tag=np.full(len(g),'prueba',object)
        tag[perm[:len(g)//2]]='train';tag[perm[len(g)//2:int(len(g)*.7)]]='validacion'
        g['conjunto']=tag;a.append(g)
    a=pd.concat(a,ignore_index=True)
    return site_ids(d).map(a.set_index('sitio').conjunto).to_numpy(),a

def stats(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[False,True]).ravel()
    return dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
        sensibilidad=float(tp/(tp+fn)) if tp+fn else None,
        especificidad=float(tn/(tn+fp)) if tn+fp else None,
        proporcion_alertas_falsas=float(fp/(tp+fp)) if tp+fp else None,
        exactitud_balanceada=float(balanced_accuracy_score(y,p)))

def detect_communities(w):
    graph=nx.from_scipy_sparse_array(csr_matrix(w))
    if graph.number_of_edges():
        sets=nx.community.louvain_communities(graph,weight='weight',seed=42,resolution=1)
        modularity=float(nx.community.modularity(graph,sets,weight='weight'))
    else:sets=[{i} for i in range(len(w))];modularity=None
    sets=sorted(sets,key=lambda x:min(x));labels=np.zeros(len(w),int)
    for k,s in enumerate(sets):labels[list(s)]=k
    return labels,dict(nodos=len(w),arcos=graph.number_of_edges(),
        densidad=float(nx.density(graph)),comunidades=len(sets),modularidad=modularity,
        aislados=int(sum(v==0 for _,v in graph.degree())))

def assign(w,labels,prefix):
    # Los identificadores son categorías nominales, no magnitudes.
    scores=np.column_stack([w[:,labels==k].sum(axis=1) for k in np.unique(labels)])
    labels_out=np.unique(labels)[scores.argmax(axis=1)]
    return np.where(scores.max(axis=1)>0,[prefix+str(k) for k in labels_out],prefix+'sin_conexion')

class SimilarityLayer:
    def fit(self,x):
        self.pre=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
        self.x=self.pre.fit_transform(x)
        self.precision=LedoitWolf().fit(self.x).precision_
        dist=cdist(self.x,self.x,'mahalanobis',VI=self.precision)
        np.fill_diagonal(dist,0);self.mean=dist.sum(axis=1)/(len(dist)-1)
        self.dmax=float(dist.max())
        w=np.ones_like(dist) if self.dmax==0 else np.clip(1-dist/self.dmax,0,1)
        keep=(dist<self.mean[:,None])&(dist<self.mean[None,:])&(w>=.3)
        np.fill_diagonal(keep,False);w=np.where(keep,w,0).astype('float32')
        self.labels,self.report=detect_communities(w)
        self.report['distancia_maxima']=self.dmax
        self.report['fraccion_arcos_peso_igual_1']=float((w==1).sum()/max(1,(w>0).sum()))
        return w

    def weights(self,x):
        dist=cdist(self.pre.transform(x),self.x,'mahalanobis',VI=self.precision)
        own_mean=dist.mean(axis=1)
        w=np.ones_like(dist) if self.dmax==0 else np.clip(1-dist/self.dmax,0,1)
        return np.where((dist<own_mean[:,None])&(dist<self.mean[None,:])&(w>=.3),w,0).astype('float32')

def bank():
    path=ROOT/'simulacion_diaria_original'
    a=np.load(path/'aptitud_diaria.npz');m=np.load(path/'respuestas_medias.npz')
    meta=json.loads((path/'metadata.json').read_text());ids={str(v):i for i,v in enumerate(a['ids'])}
    xy={f"{p['x']:.3f}|{p['y']:.3f}":ids[str(p['id_punto'])] for p in meta['puntos']}
    return dict(xy=xy,dates=pd.DatetimeIndex(a['fechas']),
        arrays={k:a[k] for k in a.files if k not in ['ids','fechas']},
        means={k:m[k] for k in m.files})

def layer_features(d,b):
    d=d.reset_index(drop=True);p=periods(d).iloc[0]
    if periods(d).nunique()!=1:raise ValueError('Se requiere un periodo por bloque.')
    date=pd.Timestamp(p[:7]+'-'+('01' if p[-1]=='1' else '16'))
    if date not in b['dates']:raise ValueError('Periodo fuera del año simulado disponible.')
    if 'Fecha_muestreo_exacta' in d and d.Fecha_muestreo_exacta.notna().any():
        raise ValueError('Este perfil usa bloques quincenales; no sustituye fechas exactas por quincenas silenciosamente.')
    day=b['dates'].get_loc(date);ids=site_ids(d).map(b['xy'])
    if ids.isna().any():raise ValueError('Ubicación sin simulación original; no se inventan valores.')
    ids=ids.to_numpy(int)
    aspect=np.radians(d.Orientacion_ladera_MDE_grados)
    topo=np.column_stack([d[TERRAIN],np.sin(aspect),np.cos(aspect)])
    sim=np.column_stack([a[day,ids] for a in list(b['arrays'].values())+list(b['means'].values())])
    prev=None if day<15 else np.column_stack([
        (b['arrays'][s+'_respuesta_realizacion_1'][day-15:day,ids]>=.9).T
        for s in ['germinacion','apresorio']]).astype(float)
    location=d[['Coordenada_X','Coordenada_Y','Altitud_MDE_contexto_m']].to_numpy()
    urban=d[GEO+['Dist_edificio_cartografiado_m']].to_numpy()
    return dict(topografia=topo,simulada=sim,condiciones15=prev,ubicacion=location,urbanistica=urban)

def fit_graphs(d,split,b,out):
    membership=pd.DataFrame(index=d.index);blocks={};reports=[];static={};static_w={}
    for period,ix in d.groupby(periods(d),sort=True).groups.items():
        local=d.loc[ix].copy();local['_sitio']=site_ids(local)
        u=local.drop_duplicates('_sitio').reset_index(drop=True)
        train_sites=set(site_ids(d.loc[split=='train']))
        fit=u._sitio.isin(train_sites).to_numpy();train=u.loc[fit].reset_index(drop=True)
        rest=u.loc[~fit].reset_index(drop=True);xf=layer_features(train,b)
        xr=layer_features(rest,b);models={};ws=[];wr=[];labels_u=pd.DataFrame({'sitio':u._sitio})
        for name in LAYERS:
            prefix=(name+'_' if name in ['topografia','ubicacion','urbanistica'] else name+'_'+period+'_')
            if xf[name] is None:
                labels_u['comunidad_'+name]=prefix+'ventana_incompleta';continue
            if name in static:
                model=static[name];w=static_w[name]
            else:
                model=SimilarityLayer();w=model.fit(xf[name])
                if name in ['topografia','ubicacion','urbanistica']:static[name]=model;static_w[name]=w
                save_npz(out/('arcos_'+prefix.rstrip('_')+'.npz'),csr_matrix(w))
            r=model.weights(xr[name]);models[name]=model;ws.append(w);wr.append(r)
            labels_u.loc[fit,'comunidad_'+name]=[prefix+str(k) for k in model.labels]
            labels_u.loc[~fit,'comunidad_'+name]=assign(r,model.labels,prefix)
            reports.append(dict(periodo=period,capa=name,**model.report))
        w=np.mean(ws,axis=0);w[w<.3]=0;np.fill_diagonal(w,0)
        labels,report=detect_communities(w);r=np.mean(wr,axis=0);r[r<.3]=0
        prefix='fusion_'+period+'_'
        labels_u.loc[fit,'comunidad_fusion']= [prefix+str(k) for k in labels]
        labels_u.loc[~fit,'comunidad_fusion']=assign(r,labels,prefix)
        labels_u['ventana15_incompleta']=int(xf['condiciones15'] is None)
        labels_u['fuerza_fusion']=0.;labels_u.loc[fit,'fuerza_fusion']=w.sum(axis=1)/max(1,len(w)-1)
        labels_u.loc[~fit,'fuerza_fusion']=r.sum(axis=1)/len(w)
        reports.append(dict(periodo=period,capa='fusion',**report))
        save_npz(out/('arcos_fusion_'+period+'.npz'),csr_matrix(w))
        csv(train[['Municipio','Parcela','Coordenada_X','Coordenada_Y','_sitio']],out/('nodos_'+period+'.csv'))
        mapping=labels_u.set_index('sitio')
        for c in mapping:membership.loc[ix,c]=local._sitio.map(mapping[c]).to_numpy()
        blocks[period]=dict(models=models,labels=labels,train_ids=train._sitio.tolist())
        print('Redes',period,report,flush=True)
    csv(pd.DataFrame(reports),out/'redes_resumen.csv')
    return membership,blocks

def graph_transform(d,b,blocks):
    out=pd.DataFrame(index=d.index)
    for period,ix in d.groupby(periods(d),sort=True).groups.items():
        if period not in blocks:raise ValueError('No hay comunidades de referencia para este periodo. Actualice las redes con datos nuevos.')
        block=blocks[period];local=d.loc[ix];u=local.assign(_sitio=site_ids(local)).drop_duplicates('_sitio').reset_index(drop=True)
        xf=layer_features(u,b);weights=[];o=pd.DataFrame({'sitio':u._sitio})
        for name in LAYERS:
            prefix=(name+'_' if name in ['topografia','ubicacion','urbanistica'] else name+'_'+period+'_')
            if name not in block['models']:
                o['comunidad_'+name]=prefix+'ventana_incompleta';continue
            model=block['models'][name];w=model.weights(xf[name]);weights.append(w)
            o['comunidad_'+name]=assign(w,model.labels,prefix)
            # Sitios conocidos conservan la comunidad train; no se reinsertan.
            ref=dict(zip(block['train_ids'],model.labels));known=u._sitio.isin(ref)
            o.loc[known,'comunidad_'+name]=[prefix+str(ref[s]) for s in u.loc[known,'_sitio']]
        w=np.mean(weights,axis=0);w[w<.3]=0;prefix='fusion_'+period+'_'
        o['comunidad_fusion']=assign(w,block['labels'],prefix)
        ref=dict(zip(block['train_ids'],block['labels']));known=u._sitio.isin(ref)
        o.loc[known,'comunidad_fusion']=[prefix+str(ref[s]) for s in u.loc[known,'_sitio']]
        o['fuerza_fusion']=w.sum(axis=1)/w.shape[1]
        o['ventana15_incompleta']=int(xf['condiciones15'] is None)
        for c in o.columns[1:]:out.loc[ix,c]=site_ids(local).map(o.set_index('sitio')[c]).to_numpy()
    return out

def tabular(d,m,arm='combinadas'):
    base=inputs(d,None).reset_index(drop=True)
    m=m.reset_index(drop=True)
    if arm=='originales':return base
    if arm in ['comunidades','tres_redes','fusion']:base=base[STAGES+GEO+['mes_sen','mes_cos']+['Temperatura promedio','Punto_de_rocio','Temperatura','Humedad_especifica','Humedad_relativa','Precipitacion']]
    cat=m[[c for c in m if c.startswith('comunidad_')]]
    if arm=='tres_redes':cat=cat[['comunidad_topografia','comunidad_simulada','comunidad_condiciones15']]
    if arm=='fusion':cat=cat[['comunidad_fusion']]
    return pd.concat([base,cat,m[['ventana15_incompleta']]],axis=1)

def preprocessor(f):
    cat=[c for c in f if c.startswith('comunidad_')];num=[c for c in f if c not in cat]
    transforms=[('numericas',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num)]
    if cat:transforms.append(('categorias',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat))
    return ColumnTransformer(transforms,sparse_threshold=0)

def forest(seed=42):
    return RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=15,
        class_weight='balanced_subsample',n_jobs=2,random_state=seed)

def fit_neural(x,y,layers,seed):
    if len(np.unique(y))<2:raise ValueError('La ruta correctora no contiene ambas clases; no se puede entrenar.')
    pre=make_pipeline(SimpleImputer(strategy='median'),StandardScaler());z=pre.fit_transform(x)
    model=MLPClassifier(hidden_layer_sizes=layers,alpha=1,max_iter=650,
        early_stopping=False,random_state=seed,tol=1e-4)
    counts=np.bincount(y.astype(int),minlength=2);w=len(y)/(2*counts[y.astype(int)])
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');model.fit(z,y,sample_weight=w)
    return dict(pre=pre,model=model,advertencias=[str(v.message) for v in notices])

def neural_prob(model,x):return model['model'].predict_proba(model['pre'].transform(x))[:,1]

def reg_input(name,z,d):return np.column_stack([z,cycle(d)]) if name=='ciclico' else z

def reg_predict(model,name,z,d):
    p=model.predict(reg_input(name,z,d))
    if name=='log1p_lineal':p=np.expm1(np.clip(p,-30,30))
    return np.maximum(1,p)

class PositiveRepresentation:
    """ACP95 sólo numéricas; categorías nominales completas, sin orden artificial."""
    def fit(self,f):
        self.cat=[c for c in f if c.startswith('comunidad_')]
        self.num=[c for c in f if c not in self.cat]
        self.numeric=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PCA(n_components=.95,svd_solver='full'))
        self.numeric.fit(f[self.num]);self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(f[self.cat])
        self.n_components_=self.numeric[-1].n_components_
        return self
    def transform(self,f):
        return np.column_stack([self.numeric.transform(f[self.num]),self.encoder.transform(f[self.cat])])

def fit_regressor(name,z,d,y,n_components=None):
    if name=='genetica_3_niveles':model=SymbolicRegressor3(population=256,generations=40,seed=42)
    elif name.startswith('red_'):model=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(6,9,18,36,36),alpha=1,max_iter=1000,early_stopping=False,random_state=42))
    elif name.startswith('polinomio_'):
        # Interacciones polinómicas de variables numéricas; efectos de comunidad
        # aditivos. Evita potencias de identificadores y expansión de dummies.
        poly=ColumnTransformer([('polinomio',PolynomialFeatures(int(name[-1]),include_bias=False),list(range(n_components or z.shape[1])))],remainder='passthrough',sparse_threshold=0)
        model=make_pipeline(poly,StandardScaler(),Ridge(alpha=100))
    else:model=make_pipeline(StandardScaler(),Ridge(alpha=10))
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');model.fit(reg_input(name,z,d),np.log1p(y) if name=='log1p_lineal' else y)
    return model,[str(v.message) for v in notices]

def run():
    threadpool_limits(2);OUT.mkdir(exist_ok=False)
    def event(role,action,detail):
        row=dict(fecha=datetime.now(timezone.utc).isoformat(),responsable=role,accion=action,evidencia=detail)
        with (OUT/'eventos.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(role,action,str(detail)[:500],flush=True)
    d=assemble(ROOT/'geografia_roya/base_geografica_exploratoria.csv',ROOT/'barreras_naturales/analisis/relieve_y_estructuras_por_sitio.csv')
    split,assignment=partition(d);csv(assignment,OUT/'particion_ubicaciones.csv')
    masks={s:split==s for s in ['train','validacion','prueba']};tr=masks['train'];va=masks['validacion'];te=masks['prueba']
    y=d[TARGET].to_numpy(float);yb=y>0;b=bank()
    event('datos','particion_por_coordenadas',{s:dict(registros=int(m.sum()),positivos=int(yb[m].sum()),sitios=int(site_ids(d[m]).nunique())) for s,m in masks.items()})
    m,blocks=fit_graphs(d,split,b,OUT)
    csv(pd.concat([d[['fila_excel','Municipio','Parcela','Mes_registrado','Quincena','Coordenada_X','Coordenada_Y']],m,pd.Series(split,name='conjunto')],axis=1),OUT/'comunidades_por_observacion.csv')
    arms={};comparison=[]
    for arm in ['originales','comunidades','combinadas','tres_redes','fusion']:
        f=tabular(d,m,arm);pre=preprocessor(f);x=pre.fit_transform(f[tr]);v=pre.transform(f[va]);t=pre.transform(f[te]);model=forest().fit(x,yb[tr]);pv=model.predict_proba(v)[:,1];pt=model.predict_proba(t)[:,1]
        thresholds=np.arange(.1,.901,.025);threshold=float(max(thresholds,key=lambda c:balanced_accuracy_score(yb[va],pv>=c)))
        arms[arm]=dict(pre=pre,model=model,threshold=threshold)
        for s,prob in [('validacion',pv),('prueba',pt)]:comparison.append(dict(variante=arm,conjunto=s,**stats(yb[masks[s]],prob>=threshold)))
        event('modelacion','bosque_'+arm,comparison[-2:])
    # La rama pedida se evalúa completa aunque otra representación gane.
    arm='combinadas';f=tabular(d,m,arm);selected=arms[arm];pre=selected['pre'];model=selected['model'];threshold=selected['threshold']
    x=pre.transform(f[tr]);v=pre.transform(f[va]);t=pre.transform(f[te]);allx=pre.transform(f)
    groups=site_ids(d[tr]).to_numpy();folds=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    foldid=np.zeros(tr.sum(),int);oof=np.zeros(tr.sum())
    for k,(a,c) in enumerate(folds.split(x,yb[tr],groups)):
        fit=forest(42+k).fit(x[a],yb[tr][a]);oof[c]=fit.predict_proba(x[c])[:,1];foldid[c]=k
        event('validacion','OOF_bosque',dict(fold=k,registros=len(c)))
    # El umbral que define los ejemplos correctores es fijo 0.5 (no validación).
    negative=oof<.5;rescue_x=np.column_stack([x,oof]);rescue=fit_neural(rescue_x[negative],yb[tr][negative],(6,9,12),42)
    rescue_oof=np.zeros(tr.sum())
    for k in range(5):
        a=negative&(foldid!=k);c=negative&(foldid==k)
        net=fit_neural(rescue_x[a],yb[tr][a],(6,9,12),50+k)
        rescue_oof[c]=neural_prob(net,rescue_x[c])
    candidate=(oof>=.5)|(rescue_oof>=.5)
    pheno=d.loc[tr,STAGES].to_numpy();confirm=fit_neural(pheno[candidate],yb[tr][candidate],(10,5,10,5,10,5),42)
    probabilities={}
    for s in ['validacion','prueba']:
        mask=masks[s];xx=allx[mask];p=model.predict_proba(xx)[:,1]
        r=neural_prob(rescue,np.column_stack([xx,p]));c=neural_prob(confirm,d.loc[mask,STAGES].to_numpy())
        probabilities[s]=(p,r,c)
    pv,rv,cv=probabilities['validacion'];baseline=pv>=threshold;baseline_fn=stats(yb[va],baseline)['FN']
    policies=[]
    for rcut in np.r_[np.arange(.05,.951,.05),1.01]:
        recovered=baseline|((~baseline)&(rv>=rcut))
        for ccut in np.r_[0.,np.arange(.05,.951,.05)]:
            final=recovered&(cv>=ccut);score=stats(yb[va],final)
            policies.append(dict(umbral_rescate=float(rcut),umbral_filtro=float(ccut),**score))
    eligible=[q for q in policies if q['FN']<=baseline_fn]
    policy=max(eligible,key=lambda q:(q['exactitud_balanceada'],-q['FP'],-q['FN']))
    csv(pd.DataFrame(policies),OUT/'politicas_validacion.csv')
    stages=[];final_masks={};prediction=[]
    for s,(p,r,c) in probabilities.items():
        mask=masks[s];first=p>=threshold;rescued=first|((~first)&(r>=policy['umbral_rescate']));final=rescued&(c>=policy['umbral_filtro']);final_masks[s]=final
        for name,pred in [('bosque',first),('bosque_rescate',rescued),('cascada_completa',final)]:stages.append(dict(conjunto=s,etapa=name,**stats(yb[mask],pred)))
        table=d.loc[mask,['fila_excel','Municipio','Parcela','Mes_registrado','Quincena','Coordenada_X','Coordenada_Y',TARGET]].copy()
        table['conjunto']=s;table['p_bosque']=p;table['p_rescate_condicional']=r;table['p_filtro_fenologia_condicional']=c
        table['bosque_positivo']=first;table['rescatado']=(~first)&rescued;table['presencia_predicha']=final
        prediction.append(table)
    csv(pd.DataFrame(comparison),OUT/'comparacion_representaciones.csv');csv(pd.DataFrame(stages),OUT/'comparacion_etapas.csv')
    csv(pd.DataFrame({'fila_excel':d.loc[tr,'fila_excel'],'fold':foldid,'prob_bosque_OOF':oof,'prob_rescate_OOF':rescue_oof,'candidato_filtro':candidate,'presencia_real':yb[tr]}),OUT/'entrenamiento_correctores_OOF.csv')
    event('estadistica','politica_elegida_validacion',policy)
    pos=tr&yb;reg_representation=PositiveRepresentation().fit(f[pos]);ncomp=reg_representation.n_components_
    z=reg_representation.transform(f);regressors={};rows=[]
    # Siete familias sobre positivos train; selección por MAE total validación.
    for name in REGRESSIONS:
        reg,notices=fit_regressor(name,z[pos],d[pos],y[pos],ncomp);regressors[name]=reg
        for s in ['validacion','prueba']:
            mask=masks[s];q=reg_predict(reg,name,z[mask],d[mask]);out=np.where(final_masks[s],q,0)
            rows.append(dict(modelo=name,conjunto=s,MAE=float(mean_absolute_error(y[mask],out)),RMSE=float(np.sqrt(mean_squared_error(y[mask],out))),
                MAE_positivos_presencia_conocida=float(mean_absolute_error(y[mask][yb[mask]],q[yb[mask]])),advertencias=' | '.join(notices)))
        event('modelacion','regresion_'+name,rows[-2:])
    clean=[r for r in rows if r['conjunto']=='validacion' and not r['advertencias']]
    if not clean:raise RuntimeError('Ninguna regresión convergió sin advertencias.')
    best=min(clean,key=lambda a:a['MAE'])['modelo'];csv(pd.DataFrame(rows),OUT/'comparacion_siete_regresiones.csv')
    for s,table in zip(['validacion','prueba'],prediction):
        mask=masks[s];table['incidencia_predicha']=np.where(final_masks[s],reg_predict(regressors[best],best,z[mask],d[mask]),0)
    csv(pd.concat(prediction),OUT/'predicciones.csv')
    bundle=dict(bank=b,blocks=blocks,arms=arms,arm=arm,rescue=rescue,confirm=confirm,policy=policy,
        reg_representation=reg_representation,regressors=regressors,selected_regression=best,
        lookup=d[['Municipio','Parcela','Coordenada_X','Coordenada_Y']+TERRAIN+GEO+['Orientacion_ladera_MDE_grados','Dist_edificio_cartografiado_m']].drop_duplicates())
    save_model(bundle,OUT/'modelo.joblib')
    report=dict(diseno='50/20/30 por coordenadas completas, estratificado por municipio, semilla42; mismas fechas y parcelas',
        particion={s:dict(registros=int(q.sum()),sitios=int(site_ids(d[q]).nunique()),positivos=int(yb[q].sum())) for s,q in masks.items()},
        regla='Arco si d_ij<media_i y d_ij<media_j y 1-d_ij/dmax>=0.30. Mahalanobis con LedoitWolf en train. Fusión: promedio de cinco capas, ausencias de ventana excluidas del denominador; poda final 0.30.',
        comunidades='Topografía, aptitud térmica simulada del día, vectores 15+15 binarios previos, ubicación, urbanística y fusión. Etiquetas nominales; dinámicas identificadas por periodo.',
        redes_inductivas='Asignar ubicación reservada por suma de pesos hacia comunidades train. Sin conexión: categoría explícita. Medias, dmax y covarianza nunca se ajustan con validación/prueba.',
        correccion='OOF bosque por coordenadas; rescate en negativos OOF, filtro sólo seis variables fenológicas en candidatos positivos OOF. Predicción de rescate OOF a su vez para seleccionar candidatos filtro. Las redes de rasgos no usan etiquetas ni datos de validación/prueba.',
        arquitectura_rescate=[6,9,12],arquitectura_filtro=[10,5,10,5,10,5],umbral_bosque=threshold,politica=policy,
        avisos_redes=dict(rescate=rescue['advertencias'],filtro=confirm['advertencias']),
        representaciones=comparison,etapas=stages,regresiones=rows,regresion_elegida=best,
        ACP_regresion=dict(componentes=int(ncomp),varianza_explicada=float(reg_representation.numeric[-1].explained_variance_ratio_.sum()),categorias=len(reg_representation.encoder.get_feature_names_out()),motivo='ACP95 sobre numéricas; categorías nominales one-hot adicionales. Polinomios sólo sobre componentes, comunidades con efectos aditivos.'),
        pendientes=['Ceros de muestreos17–24 y unidad de incidencia sin confirmar.','Exploratorio: se ha consultado repetidamente la prueba histórica; no valida futuro ni parcelas nuevas.',
          'EPSG32614 provisional; Sultepec parcelas1/2 comparten coordenadas, mantenidas en el mismo conjunto.',
          'No hay inventario verificado de centros urbanos ni todas las fuentes de agua: se usan zonas residenciales y cauces cartografiados.',
          'La simulación es aptitud térmica, no clima completo ni infección confirmada. Umbral0.9 heredado sin validación biológica.',
          'Sin fechas exactas: referencia conservadora primer día de quincena; primer periodo sin15d previos no se rellena con cero.',
          'Las comunidades no son estados biológicos. No se fuerza número de comunidades ni se les asigna orden.',
          'La etiqueta de comunidad por sí sola no constituye una probabilidad calibrada. Las salidas neuronales condicionales no se multiplican.',
          'OOF supervisado usa rasgos de redes ajustadas sobre covariables de todo train; validación y prueba son inductivas y no participan.'])
    (OUT/'resultados.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    event('orquestador','experimento_finalizado',dict(regresion=best,etapas=stages))
    make_report(report)
    return report

def enrich(bundle,d):
    needed=TERRAIN+GEO+['Orientacion_ladera_MDE_grados','Dist_edificio_cartografiado_m'];missing=[c for c in needed if c not in d]
    if not missing:return d
    left=d.copy();right=bundle['lookup'].copy()
    left['_sitio']=site_ids(left);right['_sitio']=site_ids(right)
    result=left.merge(right[['_sitio']+missing].drop_duplicates('_sitio'),on='_sitio',how='left',validate='many_to_one',sort=False).drop(columns='_sitio')
    if result[missing].isna().any().any():raise ValueError('Faltan características geográficas para estos sitios.')
    return result

def predict(bundle,d,stage='completa'):
    d=enrich(bundle,d).reset_index(drop=True);m=graph_transform(d,bundle['bank'],bundle['blocks']);f=tabular(d,m,bundle['arm'])
    selected=bundle['arms'][bundle['arm']];x=selected['pre'].transform(f);p=selected['model'].predict_proba(x)[:,1];first=p>=selected['threshold']
    r=neural_prob(bundle['rescue'],np.column_stack([x,p]));c=neural_prob(bundle['confirm'],d[STAGES].to_numpy())
    rescued=first|((~first)&(r>=bundle['policy']['umbral_rescate']));positive=rescued&(c>=bundle['policy']['umbral_filtro'])
    if stage=='bosque':positive=first
    elif stage=='filtro05':positive=rescued&(c>=.5)
    z=bundle['reg_representation'].transform(f);name=bundle['selected_regression']
    q=np.where(positive,reg_predict(bundle['regressors'][name],name,z,d),0)
    return pd.DataFrame(dict(probabilidad_presencia=p,probabilidad_rescate_condicional=r,probabilidad_filtro_condicional=c,
        presencia_predicha=positive,grupo_final=m.comunidad_fusion,incidencia_predicha=q,estado=np.where(positive,'cantidad_estimada','sin_roya')))

def make_report(r):
    lines=['# Redes y clasificación granular de roya','',
      'Experimento ejecutado. Las comparaciones utilizan una nueva división por ubicaciones: los valores no son directamente comparables con los experimentos históricos por plantas.','',
      '## Diseño','',r['regla'],'',r['redes_inductivas'],'',r['correccion'],'',
      'Topografía y ubicación son estáticas; simulación y ventana previa son dinámicas. Los nombres de comunidades dinámicas incluyen el periodo para evitar confundir etiquetas entre redes independientes.','',
      '## Presencia por etapa','',pd.DataFrame(r['etapas']).to_markdown(index=False),'',
      '## Representaciones','',pd.DataFrame(r['representaciones']).to_markdown(index=False),'',
      '## Siete modelos de magnitud','',pd.DataFrame(r['regresiones']).drop(columns='advertencias').to_markdown(index=False),'',
      'Elegido por MAE total de validación: **'+r['regresion_elegida']+'**. El MAE total incluye errores de presencia. El error con presencia conocida se informa como diagnóstico separado.','',
      'Regresión global con comunidades como predictores. No se ajustan regresiones distintas para cada comunidad temporal: muchas tienen pocos positivos y los identificadores cambian por periodo. Se conservan los perfiles históricos por grupos.','',
      'Arquitectura de rescate: 6–9–12. Filtro fenológico: 10–5–10–5–10–5. Una salida alta del filtro no equivale a confirmación biológica. Umbral elegido: '+str(r['politica']['umbral_filtro'])+'. Si es0, el filtro permanece entrenado pero desactivado. El perfil `redes_granulares_filtro05` permite examinar explícitamente el filtro0.5, sin recomendarlo.','',
      'No se usa incidencia, planta ni identificadores de parcela como entradas. Clima real y fisiología quedan como variables numéricas; las comunidades se codifican con one-hot.','',
      'ACP de regresión: '+str(r['ACP_regresion']), '',
      'Política elegida: '+str(r['politica']),'','## Límites','']+['- '+v for v in r['pendientes']]+['',
      '## Ejecución y trazabilidad','',
      '`python orquestador_redes_granulares.py --entrenar` entrena en carpeta nueva. Sin --entrenar verifica el resultado persistido.','',
      'eventos.jsonl registra responsabilidades secuenciales del orquestador (datos, redes, modelación, estadística y verificación); no simula conversaciones entre agentes independientes.','',
      'Referencias de implementación: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html ; https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html ; https://scikit-learn.org/stable/common_pitfalls.html']
    if r.get('diagnostico_filtro05'):
        lines+=['','## Diagnóstico del filtro fenológico obligatorio','',pd.DataFrame(r['diagnostico_filtro05']).to_markdown(index=False),'',
          'Forzar0.5 reduce alarmas y aumenta fuertemente las omisiones; por eso no se recomienda.']
    if 'pureza_municipal_comunidad_fusion' in r:
        lines+=['','Pureza municipal de las comunidades fusionadas: '+f"{r['pureza_municipal_comunidad_fusion']:.2%}"+'. Los grupos reflejan fuertemente ubicación; esto no demuestra mecanismo de propagación ni causalidad.']
    if 'referencias_magnitud' in r:
        lines+=['','## Referencias sencillas para magnitud','',pd.DataFrame(r['referencias_magnitud']).to_markdown(index=False),'']
    if 'expresion_genetica' in r:
        lines+=['','Programación genética seleccionada: '+r['expresion_genetica']+'. Las variables Z están estandarizadas por el propio modelo.','',
          'Nombres de las entradas usadas: '+str(r.get('entradas_genetica',{}))+'. Que una fórmula gane entre candidatas no demuestra capacidad de pronóstico futuro.']
    (OUT/'INFORME.md').write_text('\n'.join(lines))

if __name__=='__main__':
    # Mantener una identidad de módulo importable para joblib.
    from redes_granulares_roya import run as canonical_run
    canonical_run()
