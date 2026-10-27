"""Tres grafos de co-pertenencia a regímenes; no son redes neuronales."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from scipy.sparse import csr_matrix,save_npz,load_npz
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.fusion_modal import PROJECT,simulation_path
from roya.registro import write,digest,Registry

OUT=PROJECT/'runs/grafos_dinamicos'

class Regimes:
    def fit(self,x,max_components=6,seed=42,max_fit=30000):
        x=np.asarray(x,float);good=np.isfinite(x).all(axis=1);self.input_rows_=int(len(x));x=x[good]
        if len(x)<10:raise ValueError('Datos insuficientes para regímenes')
        rng=np.random.default_rng(seed)
        if len(x)>max_fit:x=x[rng.choice(len(x),max_fit,replace=False)]
        self.n_fit_=len(x);self.scaler=StandardScaler().fit(x);z=self.scaler.transform(x);self.candidates_=[];best=None
        for k in range(1,min(max_components,len(x)//5)+1):
            g=GaussianMixture(k,covariance_type='full',reg_covar=1e-3,n_init=2,max_iter=300,random_state=seed).fit(z)
            if not g.converged_:continue
            bic=float(g.bic(z));self.candidates_.append(dict(componentes=k,BIC=bic))
            if best is None or bic<best[0]:best=(bic,g)
        if best is None:raise RuntimeError('Ninguna mezcla convergió')
        self.model=best[1];return self
    def transform(self,x):
        x=np.asarray(x,float);valid=np.isfinite(x).all(axis=1);z=np.full_like(x,np.nan);p=np.full((len(x),self.model.n_components),np.nan)
        if valid.any():z[valid]=self.scaler.transform(x[valid]);p[valid]=self.model.predict_proba(z[valid])
        return z,p,valid

def pair_probability(p,i,j):
    """P(Z_i=Z_j|x_i,x_j,modelo) bajo posteriors independientes entre nodos."""
    p=np.asarray(p,float)
    if i==j:raise ValueError('No se definen autoaristas')
    if not np.isfinite(p[[i,j]]).all():return float('nan')
    return float(np.clip(p[i]@p[j],0,1))

def adjacency(z,p,k=20,threshold=.3):
    n=len(z);valid=np.isfinite(z).all(axis=1)&np.isfinite(p).all(axis=1);ix=np.flatnonzero(valid)
    if len(ix)<2:return csr_matrix((n,n),dtype=np.float32)
    neighbors=NearestNeighbors(n_neighbors=min(k+1,len(ix)),metric='euclidean',n_jobs=1).fit(z[ix]).kneighbors(z[ix],return_distance=False)
    a=np.repeat(ix,neighbors.shape[1]);b=ix[neighbors.ravel()];w=np.einsum('ij,ij->i',p[a],p[b]);keep=(a!=b)&(w>=threshold)
    graph=csr_matrix((w[keep].astype(np.float32),(a[keep],b[keep])),shape=(n,n));graph=graph.maximum(graph.T);graph.setdiag(0);graph.eliminate_zeros();return graph

def change(previous,current):
    if previous is None:return None
    union=previous.maximum(current);intersection=previous.minimum(current)
    return float(intersection.sum()/union.sum()) if union.nnz else 1.

def summary(graph,valid,p,layer,date,previous=None):
    degree=np.diff(graph.indptr);active=np.flatnonzero(valid);sub=graph[active][:,active]
    return dict(capa=layer,fecha=str(date),nodos_total=graph.shape[0],nodos_con_datos=int(valid.sum()),aristas=graph.nnz//2,grado_medio=float(degree[valid].mean()) if valid.any() else 0.,aislados_con_datos=int(((degree==0)&valid).sum()),componentes_conexas=int(connected_components(sub,directed=False)[0]) if len(active) else 0,peso_medio=float(graph.data.mean()) if graph.nnz else None,retencion_ponderada=change(previous,graph),regimenes_activos=int(len(set(p[valid].argmax(axis=1)))) if valid.any() else 0)

def make_inputs(q,d,m):
    ids=q.g.site_ids(d);unique=sorted(ids.unique());lookup={s:i for i,s in enumerate(unique)};row_ids=ids.map(lookup).to_numpy(int);partition=pd.DataFrame({'node':row_ids,'split':m.conjunto.to_numpy()})
    if partition.groupby('node').split.nunique().max()!=1:raise ValueError('Coordenadas cruzan particiones')
    splits=partition.groupby('node').split.first().reindex(range(len(unique))).to_numpy();nodes=pd.DataFrame({'node':range(len(unique)),'sitio_original':unique,'k_sitio':[hashlib.sha256(s.encode()).hexdigest()[:16] for s in unique],'conjunto':splits})
    cols=q.e.topo_columns();topo=d[cols].copy();topo['node']=row_ids;agg=topo.groupby('node').median().reindex(range(len(unique)))
    # Orientación se agrega en el círculo; no se promedian grados cerca de 0/360.
    for sin,cos in [('orientacion_original_sen','orientacion_original_cos'),('orientacion_MDE_sen','orientacion_MDE_cos')]:
        if sin in topo and cos in topo:
            means=topo.groupby('node')[[sin,cos]].mean().reindex(agg.index);norm=np.sqrt((means**2).sum(axis=1));agg[sin]=means[sin]/norm.replace(0,np.nan);agg[cos]=means[cos]/norm.replace(0,np.nan)
    conflict=topo.groupby('node')[cols].nunique(dropna=True).gt(1).any(axis=1);nodes['conflicto_topografia']=conflict.reindex(agg.index).to_numpy();nodes['observaciones']=pd.Series(row_ids).value_counts().reindex(agg.index).to_numpy();nodes['Coordenada_X']=agg.Coordenada_X.to_numpy();nodes['Coordenada_Y']=agg.Coordenada_Y.to_numpy()
    clim=d[q.old.c.CLIMATE+q.g.STAGES].copy();clim['node']=row_ids;clim['fecha']=q.h.reference_dates(d).to_numpy();excel=clim.groupby(['fecha','node']).median().reset_index();excel['n_observaciones']=clim.groupby(['fecha','node']).size().to_numpy()
    # Sólo clima entra al grafo Excel. Fenología se conserva como atributo del nodo, sin forzar una etapa única.
    excel_conflicts=clim.groupby(['fecha','node'])[q.old.c.CLIMATE].nunique(dropna=True).gt(1).any(axis=1)
    notes=dict(coordenadas=len(unique),filas_originales=len(d),coordenadas_topografia_conflictiva=int(conflict.sum()),pares_coordenada_fecha_clima_conflictivo=int(excel_conflicts.sum()),agregacion='Mediana por coordenada/fecha; orientación media circular. Se conservan indicadores de conflicto. No se resuelve identidad física por promedio.',fenologia='Seis banderas originales independientes conservadas como proporción entre observaciones coincidentes; atributos, no entradas del régimen climático.')
    return nodes,agg,excel,notes

def simulated(q,nodes):
    meta=json.loads((q.ROOT/'simulacion_diaria_original/metadata.json').read_text());mapping={f"{p['x']:.3f}|{p['y']:.3f}":p['id_punto'] for p in meta['puntos']}
    with np.load(simulation_path(q),allow_pickle=False) as z:
        col={str(v):i for i,v in enumerate(z['ids'])};indices=nodes.sitio_original.map(mapping).map(col)
        if indices.isna().any():raise ValueError('Sitio sin simulación')
        columns={};coverage={}
        for variable in ['temperatura','humedad_relativa','lluvia']:
            a=z[variable+'_media'][:,indices.to_numpy(int)];coverage[variable]=float(np.isfinite(a).mean())
            if coverage[variable]>=.95:columns[variable]=np.log1p(np.maximum(a,0)) if variable=='lluvia' else a
        if not columns:raise ValueError('Sin variables con cobertura temporal suficiente')
        return np.stack(list(columns.values()),axis=-1),pd.to_datetime(z['fechas']),list(columns),coverage

def run(out=None):
    out=Path(out or OUT);out.mkdir(parents=True,exist_ok=True);threadpool_limits(2);q=load();d,m,_,_=q.data();nodes,topo,excel,notes=make_inputs(q,d,m);sim,days,sim_cols,coverage=simulated(q,nodes);n=len(nodes);train=nodes.conjunto.eq('train').to_numpy();rows=[];models={};members={};scaled={};registry=Registry(out,'grafos_probabilisticos_por_coordenada_v1')
    config=json.loads((PROJECT/'configs/grafos_dinamicos.json').read_text());protocol=dict(config=config,notas=notes,nodos_por_particion=nodes.conjunto.value_counts().to_dict(),evento_arista='Probabilidad de co-pertenencia a un mismo régimen latente de la capa y tiempo, no igualdad exacta de valores continuos ni probabilidad de infección',formula='p_ij(t)=sum_c q_ic(t)*q_jc(t), con q posterior GMM. Independencia condicional de asignaciones entre nodos, sin calibración meteorológica.',estimacion='GMM de 1 a 6 componentes por BIC heurístico, ajustado sólo con coordenadas train; escala train. Una mezcla fija por capa conserva etiquetas de régimen entre fechas. No es Louvain ni una red neuronal.',capas={'topografia':'Estática, salvo nuevas mediciones físicas','simulacion':str(len(days))+' fechas diarias, variables '+','.join(sim_cols),'excel':str(excel.fecha.nunique())+' fechas observadas aproximadas; sólo clima real en aristas, fenología como atributos'},cobertura_simulada=coverage,esparsificacion='20 vecinos más próximos en características estandarizadas de la misma capa/fecha; conservar p>=0.30, unión simétrica. Se puede consultar p de cualquier par; arco omitido no significa probabilidad cero.',faltantes='Nodo sin clima completo no participa en esa fecha. Humedad simulada sin serie completa excluida, no inventada.',objetivo='No se utiliza incidencia de roya para construir nodos, pesos o regímenes. No se modifica ni reentrena el predictor anterior.',limites='Redes descriptivas con todos los nodos. Fit por sitio train y todas las fechas train: no validación temporal prospectiva. Antes de predictor: ablar rasgos inductivos sólo hacia nodos train, sin etiquetas/targets de validación/prueba.',fuentes={q.old.GEO_SOURCE:digest(q.ROOT/q.old.GEO_SOURCE),'simulacion':digest(simulation_path(q))})
    write(out/'protocolo.json',protocol);registry.send('datos','auditoria','agrupar_coordenadas_sin_objetivo',notes);nodes.to_csv(out/'nodos.csv',index=False);excel.to_csv(out/'atributos_excel.csv',index=False)
    topo_cols=[c for c in topo if topo[c].notna().mean()>=.95];x=topo[topo_cols].to_numpy();model=Regimes().fit(x[train],config['max_components'],config['seed']);z,p,valid=model.transform(x);models['topografia']=model;graph=adjacency(z,p,config['neighbors'],config['threshold']);folder=out/'topografia';folder.mkdir(exist_ok=True);save_npz(folder/'estatica.npz',graph);members['topografia']=p.astype(np.float32);scaled['topografia']=z.astype(np.float32);rows.append(summary(graph,valid,p,'topografia','estatica'));print('topografia',model.model.n_components,flush=True)
    for layer,array,dates,columns in [('simulacion',sim,days,sim_cols),('excel',None,sorted(excel.fecha.unique()),q.old.c.CLIMATE)]:
        if layer=='excel':
            array=np.full((len(dates),n,len(columns)),np.nan)
            for j,date in enumerate(dates):
                sub=excel[excel.fecha==date];values=sub[columns].to_numpy(float)
                for k,col in enumerate(columns):
                    if 'Precip' in col:values[:,k]=np.log1p(np.maximum(values[:,k],0))
                array[j,sub.node.to_numpy(int)]=values
        fit=array[:,train,:].reshape(-1,array.shape[-1]);model=Regimes().fit(fit,config['max_components'],config['seed']);models[layer]=model;ps=[];zs=[];previous=None;folder=out/layer;folder.mkdir(exist_ok=True)
        for j,date in enumerate(dates):
            z,p,valid=model.transform(array[j]);graph=adjacency(z,p,config['neighbors'],config['threshold']);label=pd.Timestamp(date).strftime('%Y-%m-%d');save_npz(folder/(label+'.npz'),graph);rows.append(summary(graph,valid,p,layer,label,previous));previous=graph;ps.append(p.astype(np.float32));zs.append(z.astype(np.float32))
            if j%60==0:print(layer,j,len(dates),flush=True)
        members[layer]=np.stack(ps);scaled[layer]=np.stack(zs);np.savez_compressed(out/(layer+'_estados.npz'),fechas=np.array([str(pd.Timestamp(t).date()) for t in dates]),probabilidades=members[layer],caracteristicas=scaled[layer],columnas=np.array(columns));print(layer,'final',len(dates),model.model.n_components,flush=True)
    np.savez_compressed(out/'topografia_estados.npz',probabilidades=members['topografia'],caracteristicas=scaled['topografia'],columnas=np.array(topo_cols));joblib.dump(models,out/'modelos_regimenes.joblib',compress=3);pd.DataFrame(rows).to_csv(out/'evolucion.csv',index=False)
    report=dict(protocolo=protocol,modelos={k:dict(componentes=v.model.n_components,n_train_completo=v.input_rows_,n_ajuste=v.n_fit_,BIC=v.candidates_) for k,v in models.items()},resumen_por_capa=json.loads(pd.DataFrame(rows).groupby('capa').agg(instantes=('fecha','size'),aristas_promedio=('aristas','mean'),retencion_media=('retencion_ponderada','mean'),aislados_promedio=('aislados_con_datos','mean')).reset_index().to_json(orient='records')))
    write(out/'resultados.json',report);registry.send('orquestador','memoria','grafos_no_prueban_mejora_predictiva',report['resumen_por_capa']);registry.close();return report

def probability(layer,date,node_i,node_j,out=None):
    if layer not in ['topografia','simulacion','excel']:raise ValueError('Capa desconocida')
    out=Path(out or OUT)
    with np.load(out/(layer+'_estados.npz'),allow_pickle=False) as z:
        p=z['probabilidades']
        if layer!='topografia':
            ix=np.flatnonzero(z['fechas']==str(date))
            if len(ix)!=1:raise ValueError('Fecha sin observación en esta capa')
            p=p[ix[0]]
        return pair_probability(p,int(node_i),int(node_j))

def snapshot(layer,date=None,out=None,threshold=.1):
    """Grafo completo por probabilidad; no interpreta cachés omitidas como ceros."""
    if layer not in ['topografia','simulacion','excel']:raise ValueError('Capa desconocida')
    root=Path(out or OUT)
    with np.load(root/(layer+'_estados.npz'),allow_pickle=False) as z:
        p=z['probabilidades']
        if layer!='topografia':
            ix=np.flatnonzero(z['fechas']==str(date))
            if len(ix)!=1:raise ValueError('Fecha sin datos')
            p=p[ix[0]]
        if not np.isfinite(p).all():raise ValueError('Faltan datos, no se reemplazan por cero')
        a=p@p.T;np.fill_diagonal(a,0);a[a<threshold]=0;return csr_matrix(a)
