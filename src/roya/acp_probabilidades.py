"""Experimento controlado:ACP + probabilidadesOOF en todos los regresores.

No modifica clasificadores ni resultados históricos. La probabilidad del BNN
no se usa como regresor porque no hay OOF anidado de ese tercer nivel.
"""
from pathlib import Path
import json,gc
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.variables import calendar
from roya.modelos.magnitud import Design,metrics,uniform_by_id
from roya.modelos.genetica_mse import GeneticMSE3
from roya.registro import Registry,write,digest

PROJECT=Path(__file__).resolve().parents[2]
METHODS=['lineal_multiple','log1p_lineal','polinomio_2','polinomio_3','armonica_multiple','genetica_regresion']
PROBS=['p_contexto','p_antecedentes','p_individual']

class StackDesign:
    def __init__(self,design,with_probabilities=True):self.design=design;self.with_probabilities=with_probabilities
    def fit(self,frame,p):
        p=validate_probabilities(p);self.prob_scale=StandardScaler().fit(p);self.polys={};self.scalers={}
        b=self.base(frame,p);self.base_names=['PC'+str(j+1) for j in range(self.design.pca.n_components_)]
        if self.with_probabilities:self.base_names+=PROBS
        for name in METHODS:
            if name in ['polinomio_2','polinomio_3']:
                self.polys[name]=PolynomialFeatures(2 if name=='polinomio_2' else 3,include_bias=False).fit(b)
            self.scalers[name]=StandardScaler().fit(self.core(name,b,frame))
        self.extra_names=self.design.bins+list(self.design.cat.get_feature_names_out())+self.design.calendar
        return self
    def base(self,frame,p):
        z=self.design.continuous(frame).astype(float)
        return np.column_stack([z,self.prob_scale.transform(validate_probabilities(p))]) if self.with_probabilities else z
    def core(self,name,b,frame):
        if name in self.polys:return self.polys[name].transform(b)
        if name=='armonica_multiple':
            day=(pd.to_datetime(frame['_fecha_referencia'])-pd.Timestamp('2024-09-01')).dt.days.to_numpy(float);a=2*np.pi*day/365;h=np.column_stack([np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a)])
            return np.column_stack([b,h,(b[:,:,None]*h[:,None,:]).reshape(len(b),-1)])
        return b
    def transform(self,name,frame,p):
        b=self.base(frame,p);core=self.scalers[name].transform(self.core(name,b,frame))
        d=self.design;extra=np.column_stack([d.bits.transform(frame[d.bins]),d.cat.transform(frame[d.cats]),frame[d.calendar].to_numpy(float)])
        return np.column_stack([core,extra])

def validate_probabilities(p):
    p=np.asarray(p,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Se requieren tres probabilidades finitas entre0y1')
    return p

def probability_data(source,d,split,family):
    train=split=='train';saved=pd.read_csv(source/('probabilidades_'+family+'.csv'));assert np.array_equal(saved.fila_excel,d.fila_excel)
    if family=='MIXTA':
        entries=joblib.load(source/'MIXTA.joblib')['entries'];oofs=[];fold=None
        for entry in entries:
            old=pd.read_csv(source/('OOF_'+entry['familia']+'.csv'));assert np.array_equal(old.fila_excel,d.loc[train,'fila_excel'])
            if fold is None:fold=old.fold.to_numpy()
            assert np.array_equal(fold,old.fold);oofs.append(old['p_'+entry['vista']].to_numpy())
        oof=np.column_stack(oofs)
    else:
        old=pd.read_csv(source/('OOF_'+family+'.csv'));assert np.array_equal(old.fila_excel,d.loc[train,'fila_excel']);oof=old[PROBS].to_numpy();fold=old.fold.to_numpy()
    p=saved[PROBS].to_numpy();p[train]=oof;validate_probabilities(p)
    return p,saved.presencia_predicha.to_numpy(bool),fold

def fit_one(q,design,frame,p,y,train,validation,gate,method,with_probs,config,out):
    filename=out/(method+'.joblib')
    if filename.exists():return joblib.load(filename)
    transform=StackDesign(design,with_probs).fit(frame.loc[train].reset_index(drop=True),p[train]);x=transform.transform(method,frame.loc[train].reset_index(drop=True),p[train]);v=transform.transform(method,frame.loc[validation].reset_index(drop=True),p[validation]);candidates=[];chosen=None;best=None
    parameters=[dict(seed=s) for s in config['gp_seeds']] if method=='genetica_regresion' else [dict(alpha=a) for a in config['ridge_alphas']]
    target=np.log1p(y[train]) if method=='log1p_lineal' else y[train]
    for params in parameters:
        model=GeneticMSE3(population=config['gp_population'],generations=config['gp_generations'],**params).fit(x,target) if method=='genetica_regresion' else Ridge(solver='lsqr',tol=1e-4,max_iter=200,**params).fit(x,target)
        pv=model.predict(v);smears=[1.]
        if method=='log1p_lineal':smears.append(float(np.mean(np.exp(np.clip(np.log1p(y[train])-model.predict(x),-20,20)))))
        for smear in smears:
            value=np.expm1(np.clip(pv,-30,30)) if method=='log1p_lineal' and smear==1. else (np.exp(np.clip(pv,-30,30))*smear-1 if method=='log1p_lineal' else pv)
            pred=np.where(gate[validation],np.maximum(1,value),0.);score=metrics(y[validation],pred,gate[validation],config['uniform_half_width']);row=dict(**params,smearing=smear,**score);candidates.append(row)
            key=(score['R2_esperado_uniforme'],-score['RMSE'])
            if best is None or key>best:best=key;chosen=dict(model=model,design=transform,smearing=smear,method=method,with_probabilities=with_probs,selection=row,features=x.shape[1],n_train=int(train.sum()),noise=config['uniform_half_width'])
    chosen['candidates']=candidates
    if method=='genetica_regresion':chosen['program']=dict(expression=chosen['model'].expression_,intercept=chosen['model'].intercept_,slope=chosen['model'].slope_,levels=chosen['model'].levels_)
    q.g.save_model(chosen,filename);del x,v;gc.collect();print('ajustado',out.parent.name,out.name,method,chosen['selection']['R2_esperado_uniforme'],flush=True)
    return chosen

def gated(model,frame,p,gate,stochastic=False,seed=42,ids=None):
    gate=np.asarray(gate,bool);result=np.zeros(len(frame))
    if gate.any():
        x=model['design'].transform(model['method'],frame.loc[gate].reset_index(drop=True),np.asarray(p)[gate]);v=model['model'].predict(x)
        if model['method']=='log1p_lineal':v=np.exp(np.clip(v,-30,30))*model['smearing']-1
        result[gate]=np.maximum(1,v)
        if stochastic:
            if ids is None:raise ValueError('Se requieren identidades para el escenario')
            result[gate]+=uniform_by_id(np.asarray(ids)[gate],seed,model['noise'])
    return result

def run(config_path=None,out=None):
    threadpool_limits(2);config=json.loads(Path(config_path or PROJECT/'configs/acp_probabilidades.json').read_text());out=Path(out or PROJECT/'runs/acp_probabilidades').resolve();out.mkdir(parents=True,exist_ok=True);registry=Registry(out,config['experiment']);q=load();d,m,w,_=q.data();split=m.conjunto.to_numpy();source=PROJECT/'runs/calendario_r2/ACP';frame=pd.concat([q.e.quantity_frame(d,m,w),calendar(d)],axis=1);y=d[q.g.TARGET].to_numpy(float);tr=(split=='train')&(y>0);va=split=='validacion';te=split=='prueba'
    sources={str(p.relative_to(q.ROOT)):digest(p) for p in [q.ROOT/q.old.GEO_SOURCE,q.ROOT/q.old.TERRAIN_SOURCE]};signature=dict(config=config,sources=sources,clases_fuentes={p.name:digest(p) for p in source.glob('*.joblib')},OOF={p.name:digest(p) for p in source.glob('OOF_*.csv')})
    if (out/'protocolo.json').exists() and json.loads((out/'protocolo.json').read_text())['huella']!=signature:raise ValueError('Cambió experimento o fuentes; use otra carpeta')
    protocol=dict(huella=signature,n_train_positivos=int(tr.sum()),particion='Mismo50/20/30 porcoordenadas; clasificadoresACPcongelados.',probabilidades='Tres salidas base; filas train sóloOOF de5folds porcoordenada. Validación/prueba con modelos ajustados sólo train.',coordinador='BNNconserva puerta; su probabilidad no entra a regresores porque faltaOOFanidado de ese nivel.',transformacion='Continuas estandarizadas yACP95% sólo11900positivos train. Probabilidades estandarizadas aparte; indicadores/categorías/calendario auxiliares comunes.',armonicas='Dos armónicos anuales y sus interacciones con cadaPC/probabilidad; polinomios completos de[PC,p] con interacciones.',GP='Regresor simbólicoMSE, árbol3niveles+escalaafín train; no confundir conGPclasificador.',seleccion='MáximoR2esperado uniforme de validación, sin elegir realizaciones de ruido; prueba sólo después de congelar.',limites='Prueba histórica reutilizada; comunidades transductivas de train conservadas; falta nueva cohorte. No interpretar como validación prospectiva.')
    write(out/'protocolo.json',protocol);design=Design('ACP').fit(frame.loc[tr].reset_index(drop=True),q);q.g.save_model(design,out/'base_ACP.joblib');valrows=[];bundles={};probs={};gates={}
    request=registry.send('orquestador','modelacion','comparacion_ACP_probabilidades',protocol)
    for family in config['families']:
        p,gate,fold=probability_data(source,d,split,family);probs[family]=p;gates[family]=gate
        groups=q.g.site_ids(d[split=='train']).to_numpy()
        for k in np.unique(fold):assert not set(groups[fold==k])&set(groups[fold!=k])
        for with_p in [False,True]:
            mode='ACP_probabilidades' if with_p else 'ACP_control';directory=out/family/mode;directory.mkdir(parents=True,exist_ok=True)
            for method in config['methods']:
                obj=fit_one(q,design,frame,p,y,tr,va,gate,method,with_p,config,directory);bundles[(family,mode,method)]=obj;pred=gated(obj,frame.loc[va].reset_index(drop=True),p[va],gate[va]);valrows.append(dict(familia=family,base=mode,modelo=method,conjunto='validacion',**metrics(y[va],pred,gate[va],config['uniform_half_width'])))
    q.g.csv(pd.DataFrame(valrows),out/'validacion.csv');choices={f:max([v for v in valrows if v['familia']==f and v['base']=='ACP_probabilidades'],key=lambda v:(v['R2_esperado_uniforme'],-v['RMSE'])) for f in config['families']};best=max(choices.values(),key=lambda v:(v['R2_esperado_uniforme'],-v['RMSE']));selection=dict(por_arquitectura=choices,mejor=best,criterio='Sólo validación; la comparación con control no elige datos de prueba.')
    write(out/'seleccion_antes_prueba.json',selection);reply=registry.send('modelacion','estadistica','seleccion_congelada',selection,reply_to=request)
    independent=pd.read_csv(out/'validacion.csv');independent=independent[independent.base=='ACP_probabilidades'].sort_values(['R2_esperado_uniforme','RMSE'],ascending=[False,True]).iloc[0];assert independent.familia==best['familia'] and independent.modelo==best['modelo'];registry.send('estadistica','auditoria','seleccion_reproducida',dict(familia=best['familia'],modelo=best['modelo']),reply_to=reply)
    tests=[]
    for family in config['families']:
        result=d[['fila_excel']].copy();result['conjunto']=split;result['incidencia_observada']=y;result['presencia_predicha']=gates[family];result['regresion_aplicada']=gates[family]
        for j,col in enumerate(PROBS):result[col]=probs[family][:,j]
        for mode in ['ACP_control','ACP_probabilidades']:
            for method in config['methods']:
                model=bundles[(family,mode,method)];pred=gated(model,frame,probs[family],gates[family]);result[mode+'__'+method]=pred;row=dict(familia=family,base=mode,modelo=method,conjunto='prueba',**metrics(y[te],pred[te],gates[family][te],config['uniform_half_width']));tp=(y[te]>0)&gates[family][te];row['R2_TP']=float(__import__('sklearn.metrics',fromlist=['r2_score']).r2_score(y[te][tp],pred[te][tp])) if tp.sum()>1 else None;tests.append(row)
        result['incidencia_predicha']=result['ACP_probabilidades__'+choices[family]['modelo']];q.g.csv(result,out/family/'predicciones.csv')
    q.g.csv(pd.DataFrame(valrows+tests),out/'comparacion.csv');contrasts=pd.DataFrame(tests).pivot(index=['familia','modelo'],columns='base',values=['R2','MAE','R2_positivos_reales']);contrasts.columns=['__'.join(c) for c in contrasts.columns];contrasts=contrasts.reset_index();contrasts['delta_R2']=contrasts.R2__ACP_probabilidades-contrasts.R2__ACP_control;q.g.csv(contrasts,out/'aporte_probabilidades.csv')
    report=dict(seleccion=selection,protocolo=protocol,componentes=int(design.pca.n_components_),varianza_ACP=float(design.pca.explained_variance_ratio_.sum()),n_train=int(tr.sum()),prueba=tests);write(out/'resultados.json',report);registry.send('auditoria','memoria','pendiente_verificacion',dict(accion='Reproducir96comparaciones, entradasOOF y puerta negativa; no promover sólo por complejidad.'));registry.close();return report

def inference_inputs(request,family,source=None):
    from roya.modelos.clasificacion import inputs
    source=Path(source or PROJECT/'runs/calendario_r2/ACP');q=load();d,_=q.old.registered_data(request);m=pd.read_csv(q.old.OUT/'comunidades_por_fila.csv').set_index('fila_excel').loc[d.fila_excel].reset_index();ref=q.old.load_reference();w=q.h.features(d,ref['quantity']['bank'],ref['humidity_history']);views=inputs(q,d,m,w);model=joblib.load(source/(family+'.joblib'))
    if family=='MIXTA':
        cols=[]
        for entry in model['entries']:
            base=joblib.load(source/(entry['familia']+'.joblib'));j=entry['indice'];cols.append(base['models'][j].predict_proba(views[j])[:,1])
        p=np.column_stack(cols)
    else:p=np.column_stack([est.predict_proba(view)[:,1] for est,view in zip(model['models'],views)])
    details=model['bnn'].predict_details(p);gate=details.probabilidad_roya.to_numpy()>=model['selection']['umbral'];frame=pd.concat([q.e.quantity_frame(d,m,w),calendar(d)],axis=1)
    return d,frame,p,gate,details

def predict(request,family='LR',method=None,with_probabilities=True,out=None,stochastic=False,seed=42):
    out=Path(out or PROJECT/'runs/acp_probabilidades');selection=json.loads((out/'seleccion_antes_prueba.json').read_text());method=method or selection['por_arquitectura'][family]['modelo'];mode='ACP_probabilidades' if with_probabilities else 'ACP_control';model=joblib.load(out/family/mode/(method+'.joblib'));d,frame,p,gate,result=inference_inputs(request,family);value=gated(model,frame,p,gate,stochastic,seed,d.fila_excel)
    result['fila_excel']=d.fila_excel.to_numpy();result['presencia_predicha']=gate;result['regresion_aplicada']=gate;result['incidencia_predicha']=value;result['modelo']=method;result['base']=mode
    for j,col in enumerate(PROBS):result[col]=p[:,j]
    return result
