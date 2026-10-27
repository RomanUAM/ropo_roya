"""Media neuronal Poisson por sitio/momento con toda la población y ruido separado."""
from pathlib import Path
import copy,json,warnings,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from roya.fusion_modal import PROJECT,BRANCHES,ModalInputs,modal_frames,score,simulation_path
from roya.variables import calendar
from roya.bridge import load
from roya.registro import write,digest,Registry

OUT=PROJECT/'runs/poisson_modal_todos'

def audit_split(q,d,m):
    """Exige ambas clases en cada partición y coordenadas exclusivas."""
    f=pd.DataFrame({'sitio':q.g.site_ids(d),'conjunto':m.conjunto.to_numpy(),'positivo':d[q.g.TARGET].to_numpy()>0})
    if f.groupby('sitio').conjunto.nunique().max()!=1:raise ValueError('Una coordenada aparece en más de una partición')
    rows=[]
    for tag,expected in [('train',.5),('validacion',.2),('prueba',.3)]:
        g=f[f.conjunto==tag];counts=g.positivo.value_counts()
        if any(counts.get(c,0)==0 for c in [False,True]):raise ValueError('Falta presencia o ausencia en '+tag)
        sites=g.groupby('sitio').positivo.agg(['min','max']);n=g.sitio.nunique();total=f.sitio.nunique()
        if n!=round(total*expected):raise ValueError('No cumple proporción por coordenadas')
        rows.append(dict(conjunto=tag,con_roya=int(counts[True]),sin_roya=int(counts[False]),sitios=n,sitios_siempre_negativos=int((~sites['max']).sum()),sitios_mixtos=int((sites['min']!=sites['max']).sum()),fraccion_sitios=n/total))
    return rows

def scenario(lam,seed=42):
    a=np.asarray(lam,float)
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Media inválida')
    return a*(1+np.random.default_rng(seed).uniform(-.05,.05,a.shape))

def network(layers,seed):
    return MLPRegressor(loss='poisson',hidden_layer_sizes=tuple(layers),solver='adam',alpha=.01,batch_size=256,learning_rate_init=.0005,max_iter=1,warm_start=True,early_stopping=False,random_state=seed)

def train_epochs(x,y,layers,epochs,seed,checkpoints=()):
    # partial_fit conserva el estado Adam entre épocas; sin partición aleatoria interna.
    model=network(layers,seed);saved={}
    for epoch in range(1,epochs+1):
        model.partial_fit(x,y)
        if epoch in checkpoints:saved[epoch]=copy.deepcopy(model)
    return model,saved

def metrics(y,p):
    return {**score(y,p),'deviance_Poisson':float(mean_poisson_deviance(y,np.maximum(p,1e-12))),'predicciones_mayores_40':int((np.asarray(p)>40).sum())}

def month_basis(d):
    c=calendar(d);mo=c.mes_calendario.astype(int).to_numpy()
    return np.column_stack([np.eye(12)[mo-1],c[['mes_sen','mes_cos']].to_numpy(float)])

def frames_with_month(q,d,m,w):
    frames=modal_frames(q,d,m,w);cal=calendar(d)
    for frame in frames.values():
        for col in ['mes_calendario','mes_sen','mes_cos']:
            if col not in frame:frame[col]=cal[col].to_numpy()
    return frames

def run(out=None):
    out=Path(out or OUT);out.mkdir(parents=True,exist_ok=True);threadpool_limits(2);q=load();d,m,w,_=q.data();split=m.conjunto.to_numpy();y=d[q.g.TARGET].to_numpy(float);audit=audit_split(q,d,m);tr=split=='train';va=split=='validacion';te=split=='prueba';frames=frames_with_month(q,d,m,w)
    config=json.loads((PROJECT/'configs/poisson_modal.json').read_text());protocol=dict(config=config,particion=audit,unidad='Porcentaje de incidencia: cuasipérdida Poisson para media, no conteos de hojas',target='E[Y_kj | X_kj, mes_j]',ceros='Todos incluidos, sin puerta de presencia',entradas={k:list(v) for k,v in frames.items()},fuentes={q.old.GEO_SOURCE:digest(q.ROOT/q.old.GEO_SOURCE),'simulacion':digest(simulation_path(q))},OOF='5 folds por coordenadas; preprocesado y redes ajustados en cada fold train',mes='Categórico 12 niveles + seno/coseno en ramas y fusión; no identificador de sitio como predictor',ruido='lambda_escenario=lambda*(1+U(-0.05,0.05)); no se altera y ni se elige semilla por R2; métrica principal sin ruido',seleccion='Checkpoint por menor deviance Poisson de validación; prueba histórica exploratoria reutilizada',salida='Media positiva no acotada de Poisson; no se recortan valores para calcular métricas; no se afirma distribución Poisson de porcentajes ni intervalos de conteos',limites='Mes observado, fecha 1/16 aproximada. HR simulada diaria incompleta; comunidades históricas conservadas. Modela medición contemporánea, no valida pronóstico futuro.')
    if (out/'protocolo.json').exists() and json.loads((out/'protocolo.json').read_text())!=protocol:raise ValueError('Cambió protocolo; use otra carpeta')
    write(out/'protocolo.json',protocol);registry=Registry(out,config['experiment']);registry.send('datos','auditoria','ambas_clases_y_sitios_exclusivos',audit)
    source=pd.read_csv(PROJECT/'runs/calendario_r2/ACP/OOF_LR.csv');assert np.array_equal(d.loc[tr,'fila_excel'],source.fila_excel)
    folds=source.fold.to_numpy();ff=np.full(len(d),-1);ff[tr]=folds;oof=np.zeros((tr.sum(),3));preds=np.zeros((len(d),3));branches={};groups=q.g.site_ids(d)
    for j,name in enumerate(BRANCHES):
        file=out/(name+'.joblib');of=out/('OOF_'+name+'.csv');frame=frames[name]
        if file.exists() and of.exists():
            obj=joblib.load(file);branches[name]=obj;oof[:,j]=pd.read_csv(of).lambda_estimada.to_numpy();preds[:,j]=obj['network'].predict(obj['preprocess'].transform(frame));continue
        for k in sorted(set(folds)):
            fit=tr&(ff!=k);hold=tr&(ff==k);assert not set(groups[fit])&set(groups[hold]);assert set(y[fit]>0)=={False,True}
            pre=ModalInputs().fit(q,frame.loc[fit]);net,_=train_epochs(pre.transform(frame.loc[fit]),y[fit],config['base_layers'][name],config['base_epochs'],config['seed']+j)
            oof[folds==k,j]=net.predict(pre.transform(frame.loc[hold]));print('OOF',name,int(k),flush=True)
        pre=ModalInputs().fit(q,frame.loc[tr]);net,_=train_epochs(pre.transform(frame.loc[tr]),y[tr],config['base_layers'][name],config['base_epochs'],config['seed']+j);obj=dict(preprocess=pre,network=net,n_train=int(tr.sum()),n_zero=int((y[tr]==0).sum()));branches[name]=obj;joblib.dump(obj,file,compress=3);preds[:,j]=net.predict(pre.transform(frame));pd.DataFrame(dict(fila_excel=d.loc[tr,'fila_excel'].to_numpy(),fold=folds,lambda_estimada=oof[:,j])).to_csv(of,index=False)
    basis=month_basis(d);meta_train=np.column_stack([np.log1p(oof),basis[tr]]);meta_all=np.column_stack([np.log1p(preds),basis]);scaler=StandardScaler().fit(meta_train);model,snapshots=train_epochs(scaler.transform(meta_train),y[tr],config['fusion_layers'],max(config['checkpoints']),config['seed']+100,config['checkpoints']);candidates=[]
    for epoch,net in snapshots.items():candidates.append(dict(epoca=epoch,**metrics(y[va],net.predict(scaler.transform(meta_all[va])))))
    chosen=min(candidates,key=lambda r:(r['deviance_Poisson'],r['RMSE']));write(out/'seleccion_antes_prueba.json',chosen);net=snapshots[chosen['epoca']];lam=net.predict(scaler.transform(meta_all));bundle=dict(branches=branches,fusion=net,scaler=scaler,protocol=protocol,selection=chosen);joblib.dump(bundle,out/'modelo.joblib',compress=3)
    # Controles: media global train, media mensual train y GLM Poisson sin redes.
    months=calendar(d).mes_calendario;monthly=pd.DataFrame({'mes':months[tr].to_numpy(),'y':y[tr]}).groupby('mes').y.mean();mean=np.full(len(d),y[tr].mean());pmonth=months.map(monthly).fillna(y[tr].mean()).to_numpy(float)
    glm=PoissonRegressor(alpha=.01,max_iter=500).fit(basis[tr],y[tr]);pg=glm.predict(basis);joblib.dump(glm,out/'control_mes.joblib',compress=3)
    result=d[['fila_excel']].copy();result['k_sitio']=q.g.site_ids(d).map(lambda s:hashlib.sha256(s.encode()).hexdigest()[:16]);result['j_momento']=q.h.reference_dates(d).to_numpy();result['mes']=months.to_numpy();result['conjunto']=split;result['cantidad_real']=y;result['lambda_estimada']=lam;result['lambda_escenario']=scenario(lam,config['seed']);result['limite_escenario_inferior']=lam*.95;result['limite_escenario_superior']=lam*1.05
    allpred={**{name:preds[:,j] for j,name in enumerate(BRANCHES)},'fusion_Poisson':lam,'media_train':mean,'media_mensual_train':pmonth,'GLM_mes':pg};rows=[]
    for name,p in allpred.items():
        assert np.isfinite(p).all() and (p>=0).all();result[name]=p
        for tag,mask in [('validacion',va),('prueba',te)]:
            for scope,subset in [('todos',mask),('positivos',mask&(y>0)),('ceros',mask&(y==0))]:rows.append(dict(modelo=name,conjunto=tag,ambito=scope,**metrics(y[subset],p[subset])))
    for tag,mask in [('validacion',va),('prueba',te)]:rows.append(dict(modelo='fusion_escenario_ruido_5pct',conjunto=tag,ambito='todos',**metrics(y[mask],result.loc[mask,'lambda_escenario'].to_numpy())))
    result.to_csv(out/'predicciones.csv.tmp',index=False);assert len(pd.read_csv(out/'predicciones.csv.tmp'))==len(d);(out/'predicciones.csv.tmp').replace(out/'predicciones.csv');pd.DataFrame(rows).to_csv(out/'comparacion.csv',index=False);pd.DataFrame(candidates).to_csv(out/'candidatos_validacion.csv',index=False);report=dict(protocolo=protocol,seleccion=chosen,resultados=rows);write(out/'resultados.json',report);registry.send('estadistica','memoria','evaluar_sin_seleccionar_ruido',chosen);registry.close();print('TERMINADO',chosen,flush=True);return report

def predict(request,seed=42,out=None):
    out=Path(out or OUT);model=joblib.load(out/'modelo.joblib');q=load();d,_=q.old.registered_data(request);m=pd.read_csv(q.old.OUT/'comunidades_por_fila.csv').set_index('fila_excel').loc[d.fila_excel].reset_index();ref=q.old.load_reference();w=q.h.features(d,ref['quantity']['bank'],ref['humidity_history']);frames=frames_with_month(q,d,m,w);p=[]
    for name in BRANCHES:
        branch=model['branches'][name];p.append(branch['network'].predict(branch['preprocess'].transform(frames[name])))
    lam=model['fusion'].predict(model['scaler'].transform(np.column_stack([np.log1p(np.column_stack(p)),month_basis(d)])))
    result=pd.DataFrame({'fila_excel':d.fila_excel.to_numpy(),'k_sitio':q.g.site_ids(d).map(lambda s:hashlib.sha256(s.encode()).hexdigest()[:16]),'j_momento':q.h.reference_dates(d).to_numpy(),'mes':calendar(d).mes_calendario.to_numpy(),'lambda_estimada':lam,'lambda_escenario':scenario(lam,seed),'limite_escenario_inferior':lam*.95,'limite_escenario_superior':lam*1.05});return result
