"""Superposición probabilística, comunidades inductivas y regresión genética."""
from pathlib import Path
import json,random
import numpy as np,pandas as pd,joblib
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from scipy.special import expit
from threadpoolctl import threadpool_limits
from roya.grafos_dinamicos import OUT as GRAPHS
from roya.fusion_modal import PROJECT
from roya.bridge import load
from roya.registro import write,Registry,digest
OUT=PROJECT/'runs/comunidades_regresion'
LAYERS=['topografia','simulacion','excel','superpuesta']


def overlay(probabilities):
    """Cada matriz contiene posteriors, no adyacencias esparsificadas."""
    return np.mean([p@p.T for p in probabilities],axis=0)


def communities(weights,train,threshold=.1,seed=42):
    import igraph as ig
    ig.set_random_number_generator(random.Random(seed));ix=np.flatnonzero(train);a=weights[np.ix_(ix,ix)].copy();np.fill_diagonal(a,0);r,c=np.where(np.triu(a>=threshold,k=1));g=ig.Graph(n=len(ix),edges=list(zip(r.tolist(),c.tolist())),directed=False)
    membership=np.array(g.community_multilevel(weights=a[r,c].astype(float).tolist()).membership,int) if len(r) else np.arange(len(ix))
    labels=np.full(len(train),-1,int);labels[ix]=membership;other=np.flatnonzero(~train)
    # Asignación inductiva: sólo promedio de vínculos retenidos hacia cada comunidad train.
    scores=[]
    for community in sorted(set(membership)):
        w=weights[np.ix_(other,ix[membership==community])];w=np.where(w>=threshold,w,0);scores.append(w.mean(axis=1))
    scores=np.column_stack(scores);eligible=scores.max(axis=1)>0;labels[other[eligible]]=scores[eligible].argmax(axis=1)
    return labels,dict(comunidades_train=int(len(set(membership))),aristas_train=int(len(r)),validacion_prueba_sin_vinculos=int((~eligible).sum()))


def align(labels,previous,train,next_id):
    values=sorted(set(labels[train]));mapping={}
    if previous is not None:
        old=sorted(set(previous[train]));counts=np.array([[np.sum((labels[train]==v)&(previous[train]==o)) for o in old] for v in values]);a,b=linear_sum_assignment(-counts)
        for i,j in zip(a,b):
            if counts[i,j]>0:mapping[values[i]]=old[j]
    for value in values:
        if value not in mapping:mapping[value]=next_id;next_id+=1
    return np.array([mapping.get(v,-1) for v in labels],int),next_id


def build_communities(out=None):
    out=Path(out or OUT);out.mkdir(parents=True,exist_ok=True);nodes=pd.read_csv(GRAPHS/'nodos.csv');train=nodes.conjunto.eq('train').to_numpy();static=np.load(GRAPHS/'topografia_estados.npz',allow_pickle=False)['probabilidades'];sim=np.load(GRAPHS/'simulacion_estados.npz',allow_pickle=False);excel=np.load(GRAPHS/'excel_estados.npz',allow_pickle=False);dates=excel['fechas'];rows=[];stats=[];previous={k:None for k in LAYERS};nextid={k:0 for k in LAYERS}
    # Topografía fija: se calcula una sola vez y conserva la misma etiqueta en todas las fechas.
    topolabels,topostats=communities(static@static.T,train);topolabels,nextid['topografia']=align(topolabels,None,train,0)
    for j,date in enumerate(dates):
        where=np.flatnonzero(sim['fechas']==date)
        if len(where)!=1:raise ValueError('Falta fecha simulada correspondiente al Excel')
        post=[static,sim['probabilidades'][where[0]],excel['probabilidades'][j]]
        if not all(np.isfinite(p).all() for p in post):raise ValueError('Faltan datos: no se interpretan como probabilidad cero')
        matrices={name:p@p.T for name,p in zip(LAYERS[:3],post)};matrices['superpuesta']=sum(matrices.values())/3
        frame=nodes[['node','k_sitio','conjunto']].copy();frame['fecha']=str(date)
        for layer,a in matrices.items():
            if layer=='topografia':labels=topolabels;info=topostats
            else:
                labels,info=communities(a,train,threshold=.1,seed=42+j);labels,nextid[layer]=align(labels,previous[layer],train,nextid[layer]);previous[layer]=labels
            frame['comunidad_'+layer]=labels.astype(str);stats.append(dict(capa=layer,fecha=str(date),**info))
        rows.append(frame);print('comunidades',j+1,len(dates),flush=True)
    labels=pd.concat(rows,ignore_index=True);labels.to_csv(out/'comunidades_por_sitio_fecha.csv.tmp',index=False);assert len(pd.read_csv(out/'comunidades_por_sitio_fecha.csv.tmp'))==len(nodes)*len(dates);(out/'comunidades_por_sitio_fecha.csv.tmp').replace(out/'comunidades_por_sitio_fecha.csv');pd.DataFrame(stats).to_csv(out/'evolucion_comunidades.csv',index=False);return labels


class GroupedLoss:
    """MSE y MAE exactos de todas las filas, agrupando patrones de entradas idénticos."""
    def __init__(self,x,y):
        self.x,self.inverse=np.unique(np.asarray(x,float),axis=0,return_inverse=True);self.values,yi=np.unique(y,return_inverse=True);hist=np.zeros((len(self.x),len(self.values)),float);np.add.at(hist,(self.inverse,yi),1);self.count=hist.sum(axis=1);self.sum=hist@self.values;self.sumsq=hist@(self.values**2);self.cum_count=hist.cumsum(axis=1);self.cum_sum=(hist*self.values).cumsum(axis=1);self.n=len(y);self.var=float(np.var(y));self.sd=float(np.std(y))
    def evaluate(self,p):
        p=np.asarray(p,float)
        if p.ndim==1:p=p[:,None]
        mse=(self.count[:,None]*p*p-2*self.sum[:,None]*p+self.sumsq[:,None]).sum(axis=0)/self.n
        ix=np.searchsorted(self.values,p,side='right')-1;valid=ix>=0;ix=np.maximum(ix,0);r=np.arange(len(p))[:,None];c=np.where(valid,self.cum_count[r,ix],0);s=np.where(valid,self.cum_sum[r,ix],0);mae=(p*(2*c-self.count[:,None])+self.sum[:,None]-2*s).sum(axis=0)/self.n
        r2=1-mse/self.var;objective=mse/self.var+mae/self.sd
        return r2,mae,objective


def link(eta,kind):
    if kind=='logistica':return 40*expit(eta)
    if kind=='gompertz':return 40*np.exp(-np.exp(np.clip(-eta,-30,30)))
    if kind=='hill':
        a=np.logaddexp(0,eta);return 40*(a*a)/(1+a*a)
    raise ValueError('Enlace desconocido')


def inverse_link(y,kind):
    t=np.clip(np.asarray(y)/40,1e-4,1-1e-4)
    if kind=='logistica':return np.log(t/(1-t))
    if kind=='gompertz':return -np.log(-np.log(t))
    a=np.sqrt(t/(1-t));return np.where(a>30,a,np.log(np.expm1(np.minimum(a,30))))


def nondominated(r2,mae):
    return np.array([not np.any((r2>=r2[i])&(mae<=mae[i])&((r2>r2[i])|(mae<mae[i]))) for i in range(len(r2))])


def genetic_fit(x,y,kind,population=200,generations=200,seed=42):
    loss=GroupedLoss(x,y);ridge=Ridge(alpha=10,fit_intercept=False).fit(x,inverse_link(y,kind)).coef_;rng=np.random.default_rng(seed);pop=ridge+rng.normal(0,.2,(population,x.shape[1]));pop[0]=ridge;history=[];archive=[]
    for generation in range(generations+1):
        r2,mae,objective=loss.evaluate(link(loss.x@pop.T,kind));order=np.argsort(objective);pop=pop[order];r2=r2[order];mae=mae[order];objective=objective[order];history.append(dict(generacion=generation,R2=float(r2[0]),MAE=float(mae[0]),J=float(objective[0])))
        if generation%20==0 or generation==generations:
            mask=nondominated(r2,mae);archive.extend(pop[mask].copy());print('GA',kind,generation,'patrones',len(loss.x),'R2',float(r2[0]),'MAE',float(mae[0]),flush=True)
        if generation==generations:break
        # Elitismo 10%, torneo de 3, cruce aritmético y mutación gaussiana decreciente.
        elite=pop[:max(2,population//10)].copy();size=population-len(elite);a=rng.integers(population,size=(size,3)).min(axis=1);b=rng.integers(population,size=(size,3)).min(axis=1);mix=rng.uniform(-.1,1.1,(size,1));children=mix*pop[a]+(1-mix)*pop[b];mutation=rng.random(children.shape)<.12;children+=mutation*rng.normal(0,.12*(1-generation/(generations+1))+.01,children.shape);pop=np.r_[elite,children]
    archive=np.unique(np.vstack(archive),axis=0);r2,mae,_=loss.evaluate(link(loss.x@archive.T,kind));mask=nondominated(r2,mae);return archive[mask],history,dict(patrones=len(loss.x),filas=len(y),variables=x.shape[1])


def design_frame(q,d,labels):
    nodes=pd.read_csv(GRAPHS/'nodos.csv');mapping=dict(zip(nodes.sitio_original,nodes.node));f=pd.DataFrame({'node':q.g.site_ids(d).map(mapping),'fecha':pd.to_datetime(q.h.reference_dates(d)).dt.strftime('%Y-%m-%d')});right=labels.copy();right['fecha']=right.fecha.astype(str);f=f.merge(right,on=['node','fecha'],how='left',validate='many_to_one',sort=False)
    if f['comunidad_superpuesta'].isna().any():raise ValueError('Sitio/fecha sin comunidades')
    for col in ['comunidad_'+k for k in LAYERS]:f[col]=f[col].astype(str)
    f['mes']=pd.to_datetime(f.fecha).dt.month.astype(str);f['quincena']=d.Quincena.astype(str).to_numpy();return f[['comunidad_'+k for k in LAYERS]+['mes','quincena']]


def run(out=None):
    out=Path(out or OUT);out.mkdir(parents=True,exist_ok=True);threadpool_limits(2);config=json.loads((PROJECT/'configs/comunidades_regresion.json').read_text());labels_path=out/'comunidades_por_sitio_fecha.csv'
    labels=pd.read_csv(labels_path) if labels_path.exists() else build_communities(out);q=load();d,m,_,_=q.data();f=design_frame(q,d,labels);y=d[q.g.TARGET].to_numpy(float);split=m.conjunto.to_numpy();tr=split=='train';va=split=='validacion';te=split=='prueba';encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(f.loc[tr]);x=encoder.transform(f);x=np.column_stack([np.ones(len(x)),x]);rows=[];selection={};models={};result=d[['fila_excel']].copy();result['conjunto']=split;result['cantidad_real']=y
    protocol=dict(config=config,grafo='Promedio aritmético de probabilidades de tres capas en la MISMA fecha; descartar sólo promedios<0.10. Se recalculan todos los pares desde posteriors, no se toma arco esparso omitido como cero.',comunidades='Louvain ponderado sólo sobre 1,100 coordenadas train; validación/prueba se asignan por media de vínculos >=0.1 a comunidades train. Alineación temporal por máximo solapamiento/Hungarian sólo train; topografía fija.',entradas='Comunidades categóricas de tres capas y superpuesta, mes y quincena; one-hot train, sin códigos ordinales ni incidencia vecina.',objetivo='Porcentaje 0..40. Todas las filas y todos los ceros; misma partición 50/20/30 por sitio.',regresiones=['40*sigmoid(B beta)','40*exp(-exp(-B beta))','40*softplus(B beta)^2/(1+softplus(B beta)^2)'],genetico='Población 200, 200 generaciones. J=(1-R2_train)+MAE_train/SD_train; archivo no dominado en R2 y MAE. No se garantiza óptimo global ni ambos óptimos a la vez.',seleccion='Evaluar frontera train en validación con J_val=(1-R2_val)+MAE_val/SD_val. Congelar selección antes de prueba.',control='Promedio mensual train; media global train; referencia Poisson anterior como comparación histórica. Test reutilizado: exploratorio.',limites='Probabilidades GMM de co-régimen, no contagio ni meteorología calibrada. Mes/ceros tardíos y HR simulada incompleta. Capas se ajustaron con características train de todo el año; no evaluación futura.',fuente=digest(GRAPHS/'protocolo.json'))
    write(out/'protocolo.json',protocol);registry=Registry(out,'comunidades_regresion_genetica_v1')
    for j,kind in enumerate(config['models']):
        modelpath=out/(kind+'.joblib')
        if modelpath.exists():bundle=joblib.load(modelpath);beta=bundle['coef'];chosen=bundle['seleccion']
        else:
            candidates,history,counts=genetic_fit(x[tr],y[tr],kind,config['population'],config['generations'],config['seed']+j);val=GroupedLoss(x[va],y[va]);r2,mae,joint=val.evaluate(link(val.x@candidates.T,kind));pick=int(np.argmin(joint));beta=candidates[pick];chosen=dict(R2_validacion=float(r2[pick]),MAE_validacion=float(mae[pick]),J_validacion=float(joint[pick]),candidatos_pareto=len(candidates),**counts);bundle=dict(coef=beta,encoder=encoder,kind=kind,seleccion=chosen,protocol=protocol);joblib.dump(bundle,modelpath,compress=3);pd.DataFrame(history).to_csv(out/(kind+'_evolucion_genetica.csv'),index=False);pd.DataFrame(dict(R2_validacion=r2,MAE_validacion=mae,J_validacion=joint)).to_csv(out/(kind+'_pareto_validacion.csv'),index=False)
        selection[kind]=chosen;models[kind]=bundle
    winner=min(selection,key=lambda k:selection[k]['J_validacion']);write(out/'seleccion_antes_prueba.json',dict(modelo=winner,candidatos=selection));registry.send('estadistica','auditoria','seleccion_congelada',dict(modelo=winner,candidatos=selection))
    for kind,bundle in models.items():
        p=link(x@bundle['coef'],kind);result[kind]=p
        for tag,mask in [('validacion',va),('prueba',te)]:
            for scope,sub in [('todos',mask),('positivos',mask&(y>0)),('ceros',mask&(y==0))]:rows.append(dict(modelo=kind,conjunto=tag,ambito=scope,n=int(sub.sum()),R2=float(r2_score(y[sub],p[sub])) if np.var(y[sub])>0 else None,MAE=float(mean_absolute_error(y[sub],p[sub])),RMSE=float(np.sqrt(mean_squared_error(y[sub],p[sub])))))
    for kind in ['media_train','media_mensual_train']:
        if kind=='media_train':p=np.full(len(y),y[tr].mean())
        else:
            month=f.mes;means=pd.DataFrame({'mes':month[tr].to_numpy(),'y':y[tr]}).groupby('mes').y.mean();p=month.map(means).fillna(y[tr].mean()).to_numpy()
        result[kind]=p
        for tag,mask in [('validacion',va),('prueba',te)]:rows.append(dict(modelo=kind,conjunto=tag,ambito='todos',n=int(mask.sum()),R2=float(r2_score(y[mask],p[mask])),MAE=float(mean_absolute_error(y[mask],p[mask])),RMSE=float(np.sqrt(mean_squared_error(y[mask],p[mask])))))
    result.to_csv(out/'predicciones.csv.tmp',index=False);assert len(pd.read_csv(out/'predicciones.csv.tmp'))==len(d);(out/'predicciones.csv.tmp').replace(out/'predicciones.csv');pd.DataFrame(rows).to_csv(out/'comparacion.csv',index=False);report=dict(protocolo=protocol,seleccion=winner,candidatos=selection,resultados=rows);write(out/'resultados.json',report);registry.send('auditoria','memoria','no_promover_sin_superar_controles',rows);registry.close();return report


def predict(request,kind=None,out=None):
    out=Path(out or OUT);selection=json.loads((out/'seleccion_antes_prueba.json').read_text());kind=kind or selection['modelo'];model=joblib.load(out/(kind+'.joblib'));q=load();d,_=q.old.registered_data(request);labels=pd.read_csv(out/'comunidades_por_sitio_fecha.csv');f=design_frame(q,d,labels);x=np.column_stack([np.ones(len(f)),model['encoder'].transform(f)]);p=link(x@model['coef'],kind);return pd.DataFrame({'fila_excel':d.fila_excel.to_numpy(),'incidencia_estimada':p,'modelo':kind})

def superposed_snapshot(date,threshold=.1,root=None):
    """Reconstruye la red media de tres capas para una fecha observada del Excel."""
    from scipy.sparse import csr_matrix
    root=Path(root or GRAPHS);prob=[]
    for layer in LAYERS[:3]:
        with np.load(root/(layer+'_estados.npz'),allow_pickle=False) as z:
            p=z['probabilidades']
            if layer!='topografia':
                ix=np.flatnonzero(z['fechas']==str(date))
                if len(ix)!=1:raise ValueError('No hay tres capas para esta fecha')
                p=p[ix[0]]
            if not np.isfinite(p).all():raise ValueError('Faltan datos, no son probabilidad cero')
            prob.append(p)
    a=overlay(prob);np.fill_diagonal(a,0);a[a<threshold]=0;return csr_matrix(a)
