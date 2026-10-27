"""RF/GP/LR aumentados y mezcla variacional Poisson 3/4/5 x13, hacia adelante."""
import json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from roya.bridge import load
from roya.niveles_bayes import frames,labels,base,pad_prob
from roya.fusion_modal import PROJECT,ModalInputs,BRANCHES
from roya.temporal import KEYS,augment_blocks,score
from roya.modelos.bayes_poisson import BayesianPoissonMLP,ProbabilityHistoryScaler
from roya.registro import write
OUT=PROJECT/'runs/temporal_bayes'

def fit_expert(est,x,y,family):
    if family=='RF':return est.fit(x,y)
    # Compresión exacta de duplicados: mismos datos aumentados, frecuencias como pesos.
    unique,counts=np.unique(np.column_stack([x,y]),axis=0,return_counts=True)
    return est.fit(unique[:,:-1],unique[:,-1].astype(int),sample_weight=counts)


def run(presence_only=False):
    threadpool_limits(2);OUT.mkdir(parents=True,exist_ok=True);q=load();d,m,w,_=q.data();views=frames(q,d,m,w);meta=d[KEYS+['fila_excel',q.g.TARGET]].copy();meta['fecha']=q.h.reference_dates(d).to_numpy();meta['original_index']=np.arange(len(d));meta=meta.sort_values(KEYS+['fecha']).reset_index(drop=True)
    if meta.duplicated(KEYS+['fecha']).any():raise ValueError('Serie no única')
    group=meta.groupby(KEYS,sort=False);meta['serie']=group.ngroup();meta['prev_index']=group.original_index.shift(1);meta['lag1']=group[q.g.TARGET].shift(1);meta['lag2']=group[q.g.TARGET].shift(2);meta['origen']=group.fecha.shift(1);meta=meta[meta.lag2.notna()].reset_index(drop=True);assert (meta.origen<meta.fecha).all();y=meta[q.g.TARGET].to_numpy(float);yc=labels(y);k=int(yc.max())+1;dates=np.sort(meta.fecha.unique());a=round(.6*len(dates));b=round(.8*len(dates));tr=meta.fecha.isin(dates[:a]).to_numpy();va=meta.fecha.isin(dates[a:b]).to_numpy();te=meta.fecha.isin(dates[b:]).to_numpy();config=json.loads((PROJECT/'configs/temporal_bayes.json').read_text());previd=meta.prev_index.to_numpy(int);current=meta.original_index.to_numpy(int);inputframes={}
    for name,frame in views.items():
        f=frame.iloc[current if name=='topografia' else previd].reset_index(drop=True).copy();f['roya_t_1']=meta.lag1.to_numpy();f['roya_t_2']=meta.lag2.to_numpy();f['mes_objetivo_sen']=np.sin(2*np.pi*(meta.fecha.dt.month-1)/12);f['mes_objetivo_cos']=np.cos(2*np.pi*(meta.fecha.dt.month-1)/12);inputframes[name]=f
    # Misma secuencia causal para todos. Los primeros tres momentos train inicializan expertos.
    foldblocks=[np.arange(start,min(start+2,a)) for start in range(3,a,2)];oofmask=tr&meta.fecha.isin(dates[3:a]).to_numpy();oof_pos=np.flatnonzero(oofmask);position={int(row):j for j,row in enumerate(oof_pos)};allprobs={};oofs={};aug=augment_blocks(meta.loc[tr].reset_index(drop=True),y[tr]);protocol=dict(config=config,clases=13,series=2400,fechas_elegibles=len(dates),cortes=[a,b-a,len(dates)-b],filas=[int(tr.sum()),int(va.sum()),int(te.sum())],aumento_filas=len(aug),rezagos='t-1 y t-2 medidos, mismo municipio/parcela/coordenada; se excluyen dos fechas iniciales sin historia completa.',expertos='Tres RF, tres GP ordinales y tres LR multiclase; todos nuevos, aumentados sólo en train y en cada fold temporal. Calendario, incidencia previa y datos de cada vista.',OOF='Ventanas expansivas; cada bloque OOF se predice con fechas estrictamente anteriores. Mezcladores aprenden sólo en filas OOF train.',entradas='Topografía estática y urbanismo; Excel/simulación en el origen t-1; nunca se utilizan clima real o fenología futuros. No se usan comunidades climáticas ajustadas con todo el año.',mezcladores='Variacionales: normales diagonales independientes para TODOS los pesos y sesgos; arquitecturas 3x13,4x13,5x13. Objetivo cuasi-Poisson+KL/N; estimación de incidencia porcentual, no conteos.',ajuste_bayes='Expertos aumentados; mezclador sobre OOF original sin duplicar observaciones para no aumentar artificialmente la evidencia del posterior.',incertidumbre='Cuantiles Monte Carlo de lambda bajo posterior variacional aproximado; no intervalos predictivos de observaciones ni cobertura calibrada.',mixta='Mejor vista de cada familia por log-loss OOF train; no usa prueba.',evaluacion='12 combinaciones:4arquitecturas x3redes bayesianas. Selección por MAE validación. Prueba final toda cero, R2indefinido. Se conservan ceros reales.',horizonte='Un paso con observaciones previas reales actualizadas; no pronóstico recursivo del bloque completo. Fechas aproximadas.',aumento='Bloques observados repetidos, no individuos nuevos. No se impone ley de potencias. El mismo aumento se usa en cada familia.')
    protocol['detector_temporal']='Calibrador bayesiano binario ajustado en OOF causal de los expertos aumentados; umbral por suma de tasas FPR+FNR y templado por validación. El detector histórico se conserva sólo como referencia diagnóstica, no se usa como prueba cronológica.'
    protocol['cascada']='Detector temporal y cantidad se eligen por separado en validación: detector por FPR+FNR, cantidad por MAE. Se preserva detector histórico como referencia diagnóstica.'
    protocol['escala_mezclador']='Probabilidades0..1 sin amplificar clases raras; estandarizar sólo lag1,lag2,mes_sen,mes_cos. Referencia inicial conservada; validación histórica reutilizada, comparación exploratoria.'
    write(OUT/'protocolo.json',protocol)
    from sklearn.metrics import log_loss
    for family in ['RF','LR','GP']:
        pp=[];oo=[];folder=OUT/family;folder.mkdir(exist_ok=True)
        for j,name in enumerate(BRANCHES):
            cache=folder/(name+'_probas.npz');modelpath=folder/(name+'.joblib');f=inputframes[name]
            if cache.exists() and modelpath.exists():
                with np.load(cache) as z:pp.append(z['p']);oo.append(z['oof'])
                continue
            oof=np.zeros((len(oof_pos),k))
            for z,block in enumerate(foldblocks):
                fit=meta.fecha.isin(dates[:block[0]]).to_numpy();hold=meta.fecha.isin(dates[block]).to_numpy();assert meta.loc[fit,'fecha'].max()<meta.loc[hold,'fecha'].min();ix=augment_blocks(meta.loc[fit].reset_index(drop=True),y[fit]);pre=ModalInputs().fit(q,f.loc[fit]);x=pre.transform(f.loc[fit]);est=fit_expert(base(family,42+z,config),x[ix],yc[fit][ix],family);p=pad_prob(est,pre.transform(f.loc[hold]),k);oof[[position[int(row)] for row in np.flatnonzero(hold)]]=p;print('temporal',family,name,z,flush=True)
            pre=ModalInputs().fit(q,f.loc[tr]);x=pre.transform(f.loc[tr]);est=fit_expert(base(family,42,config),x[aug],yc[tr][aug],family);p=pad_prob(est,pre.transform(f),k);joblib.dump(dict(preprocess=pre,model=est),modelpath,compress=3);np.savez_compressed(cache,p=p,oof=oof);pp.append(p);oo.append(oof)
        allprobs[family]=np.stack(pp,axis=1);oofs[family]=np.stack(oo,axis=1)
    entries=[]
    for family in ['RF','GP','LR']:
        losses=[log_loss(yc[oofmask],oofs[family][:,j],labels=np.arange(k)) for j in range(3)];j=int(np.argmin(losses));entries.append(dict(familia=family,vista=BRANCHES[j],indice=j))
    allprobs['MIXTA']=np.stack([allprobs[e['familia']][:,e['indice']] for e in entries],axis=1);oofs['MIXTA']=np.stack([oofs[e['familia']][:,e['indice']] for e in entries],axis=1);write(OUT/'entradas_mixta.json',entries)
    from roya.niveles_bayes import InverseBayes
    from sklearn.metrics import log_loss as binary_logloss
    ctx=pd.DataFrame({'mes':meta.fecha.dt.month.astype(str),'lag1_clase':labels(meta.lag1).astype(str),'lag2_clase':labels(meta.lag2).astype(str)})
    gates={};presence_rows=[]
    for fam in allprobs:
        pp=np.stack([allprobs[fam][:,:,0],1-allprobs[fam][:,:,0]],axis=-1);po=np.stack([oofs[fam][:,:,0],1-oofs[fam][:,:,0]],axis=-1);det=InverseBayes().fit(po,(y[oofmask]>0).astype(int),ctx.loc[oofmask]);options=[]
        for temperature in [0.,.1,.5,1.]:
            for weight in [0.,.5,1.]:
                pv=det.predict_proba(pp[va],ctx.loc[va],temperature,weight);options.append((binary_logloss((y[va]>0).astype(int),pv,labels=[0,1]),temperature,weight))
        _,temperature,weight=min(options);pr=det.predict_proba(pp,ctx,temperature,weight)[:,1];truth=y[va]>0;thresholds=[]
        for threshold in np.arange(.05,.951,.025):
            decision=pr[va]>=threshold;fp=int(np.sum(decision&~truth));fn=int(np.sum(~decision&truth));thresholds.append((fp/max(int((~truth).sum()),1)+fn/max(int(truth.sum()),1),fp,fn,float(threshold)))
        errors,fp,fn,threshold=min(thresholds);gates[fam]=pr>=threshold;joblib.dump(dict(model=det,temperature=temperature,context_weight=weight,threshold=threshold),OUT/(fam+'_presencia_temporal.joblib'),compress=3);decision=gates[fam][te];truth=y[te]>0;presence_rows.append(dict(familia=fam,umbral=threshold,FP_validacion=fp,FN_validacion=fn,TN=int(np.sum(~decision&~truth)),FP=int(np.sum(decision&~truth)),FN=int(np.sum(~decision&truth)),TP=int(np.sum(decision&truth))))
    pd.DataFrame(presence_rows).to_csv(OUT/'presencia_temporal.csv',index=False)
    presence_selected=min(presence_rows,key=lambda row:row['FP_validacion']/max(int((y[va]==0).sum()),1)+row['FN_validacion']/max(int((y[va]>0).sum()),1));detector_family=presence_selected['familia'];write(OUT/'seleccion_presencia_temporal.json',presence_selected)
    if presence_only:return presence_rows
    extras=np.column_stack([meta.lag1,meta.lag2,np.sin(2*np.pi*(meta.fecha.dt.month-1)/12),np.cos(2*np.pi*(meta.fecha.dt.month-1)/12)]);meta_aug=augment_blocks(meta.loc[oofmask].reset_index(drop=True),y[oofmask]);rows=[];pred={};candidates=[]
    for family in allprobs:
        full=np.column_stack([allprobs[family].reshape(len(y),-1),extras]);train=np.column_stack([oofs[family].reshape(len(oof_pos),-1),extras[oofmask]]);scaler=ProbabilityHistoryScaler().fit(train);xt=scaler.transform(train);xf=scaler.transform(full)
        for layers in [3,4,5]:
            tag=family+'_'+str(layers)+'x13';file=OUT/(tag+'.joblib')
            if file.exists() and joblib.load(file).get('input_scaling_version',0)==2:model=joblib.load(file)['model']
            else:model=BayesianPoissonMLP(layers=layers,width=13,epochs=config['bayes_epochs'],seed=42).fit(xt,y[oofmask]);joblib.dump(dict(model=model,scaler=scaler,posterior_original_data=True,input_scaling_version=2),file,compress=3)
            pv=model.posterior(xf[va],config['posterior_samples']);candidates.append(dict(modelo=tag,**score(y[va],pv['media'])));pred[tag]=model.posterior(xf[te],config['posterior_samples']);print('bayes',tag,flush=True)
    selected=min(candidates,key=lambda r:r['MAE']);write(OUT/'seleccion_antes_prueba.json',selected)
    for tag,p in pred.items():rows.append(dict(modelo=tag,conjunto='prueba',**score(y[te],p['media']),ancho_medio_lambda=float(np.mean(p['p975']-p['p025']))));rows.append(dict(modelo=tag+'_cascada',conjunto='prueba',**score(y[te],np.where(gates[detector_family][te],p['media'],0))));np.savez_compressed(OUT/(tag+'_posterior_prueba.npz'),**p,fila_excel=meta.loc[te,'fila_excel'].to_numpy())
    for name,p in [('persistencia',meta.loc[te,'lag1'].to_numpy()),('cero',np.zeros(te.sum()))]:rows.append(dict(modelo=name,conjunto='prueba',**score(y[te],p)))
    pd.DataFrame(candidates).to_csv(OUT/'validacion.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'comparacion.csv',index=False);dest=PROJECT/'reports/temporal_bayes';dest.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ['protocolo.json','seleccion_antes_prueba.json','validacion.csv','comparacion.csv','entradas_mixta.json','presencia_temporal.csv','seleccion_presencia_temporal.json']:shutil.copyfile(OUT/name,dest/name)
    (dest/'INFORME.md').write_text('# Expertos aumentados y red bayesiana temporal\n\n'+json.dumps(protocol,ensure_ascii=False,indent=2)+'\n\n## Validación\n\n'+pd.DataFrame(candidates).to_markdown(index=False)+'\n\n## Prueba\n\n'+pd.DataFrame(rows).to_markdown(index=False)+'\n\nSeleccionado exclusivamente por MAE de validación: '+selected['modelo']+'. La prueba final sin positivos no valida severidad ni calibración de la incertidumbre. Cuasiposterior por respuesta porcentual; no se afirma una ley Poisson de conteos.\n');return rows

def inference_frames(q,d,m,w):
    views=frames(q,d,m,w);meta=d[KEYS+['fila_excel',q.g.TARGET]].copy();meta['fecha']=q.h.reference_dates(d).to_numpy();meta['original_index']=np.arange(len(d));meta=meta.sort_values(KEYS+['fecha']).reset_index(drop=True);g=meta.groupby(KEYS,sort=False);meta['prev_index']=g.original_index.shift(1);meta['lag1']=g[q.g.TARGET].shift(1);meta['lag2']=g[q.g.TARGET].shift(2);meta=meta[meta.lag2.notna()].reset_index(drop=True);previd=meta.prev_index.to_numpy(int);current=meta.original_index.to_numpy(int);result={}
    for name,view in views.items():
        f=view.iloc[current if name=='topografia' else previd].reset_index(drop=True).copy();f['roya_t_1']=meta.lag1.to_numpy();f['roya_t_2']=meta.lag2.to_numpy();f['mes_objetivo_sen']=np.sin(2*np.pi*(meta.fecha.dt.month-1)/12);f['mes_objetivo_cos']=np.cos(2*np.pi*(meta.fecha.dt.month-1)/12);result[name]=f
    return meta,result

def predict(request,detector='temporal',model_name=None,samples=100):
    q=load();registered,_=q.old.registered_data(request);d,m,w,_=q.data();meta,views=inference_frames(q,d,m,w);selection=json.loads((OUT/'seleccion_antes_prueba.json').read_text());tag=model_name or selection['modelo'];family=tag.split('_')[0];bundle=joblib.load(OUT/(tag+'.joblib'));ix=pd.Index(meta.fila_excel).get_indexer(registered.fila_excel)
    if np.any(ix<0):raise ValueError('Se requieren dos observaciones previas del mismo sitio')
    def probabilities(fam,indices):
        entries=json.loads((OUT/'entradas_mixta.json').read_text()) if fam=='MIXTA' else [dict(familia=fam,vista=name) for name in BRANCHES];p=[]
        for e in entries:
            obj=joblib.load(OUT/e['familia']/(e['vista']+'.joblib'));p.append(pad_prob(obj['model'],obj['preprocess'].transform(views[e['vista']].iloc[indices]),13))
        return np.stack(p,axis=1)
    if detector is False:
        gate=np.ones(len(ix),bool);presence=np.full(len(ix),np.nan);detector_name='presencia_confirmada'
    elif detector=='historico':
        from roya.acp_probabilidades import inference_inputs
        _,_,_,gate,details=inference_inputs(request,'LR');presence=details.probabilidad_roya.to_numpy();detector_name='LR/ACP historico, diagnostico'
    else:
        fam=json.loads((OUT/'seleccion_presencia_temporal.json').read_text())['familia'];obj=joblib.load(OUT/(fam+'_presencia_temporal.joblib'));f=meta.iloc[ix];ctx=pd.DataFrame({'mes':f.fecha.dt.month.astype(str).to_numpy(),'lag1_clase':labels(f.lag1).astype(str),'lag2_clase':labels(f.lag2).astype(str)});p=probabilities(fam,ix);p2=np.stack([p[:,:,0],1-p[:,:,0]],axis=-1);presence=obj['model'].predict_proba(p2,ctx,obj['temperature'],obj['context_weight'])[:,1];gate=presence>=obj['threshold'];detector_name=fam+' temporal OOF'
    result=pd.DataFrame({'fila_excel':registered.fila_excel.to_numpy(),'presencia_predicha':gate,'probabilidad_presencia':presence,'lambda_media':np.zeros(len(ix)),'lambda_p025':np.zeros(len(ix)),'lambda_p975':np.zeros(len(ix)),'modelo_cantidad':tag,'modelo_presencia':detector_name})
    if gate.any():
        indices=ix[gate];p=probabilities(family,indices);f=meta.iloc[indices];extra=np.column_stack([f.lag1,f.lag2,np.sin(2*np.pi*(f.fecha.dt.month-1)/12),np.cos(2*np.pi*(f.fecha.dt.month-1)/12)]);x=np.column_stack([p.reshape(len(indices),-1),extra]);posterior=bundle['model'].posterior(bundle['scaler'].transform(x),samples=samples)
        for key,col in [('media','lambda_media'),('p025','lambda_p025'),('p975','lambda_p975')]:result.loc[gate,col]=posterior[key]
    return result
